#!/usr/bin/env python3
"""
Script to extract SigLIP features for HF dataset records and split into train/val sets.
This script:
1. Downloads dataset from HF
2. Extracts SigLIP features for image outputs
3. Saves images locally
4. Splits data into train/val sets
5. Creates separate JSON files with/without image outputs
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# Import the SigLIP feature extraction functions
from get_siglip2_features import get_interpolated_patch_features, load_image, extract_patch_features


class SigLIPFeatureExtractor:
    """Extract SigLIP features and manage dataset splitting."""
    
    def __init__(self, 
                 output_dir: str = "./processed_data",
                 target_num_patches: int = 576,  # 24x24 patches for 576 total
                 train_ratio: float = 0.8,
                 random_seed: int = 42):
        """
        Initialize the feature extractor.
        
        Args:
            output_dir: Directory to save processed data
            target_num_patches: Number of patches to interpolate to
            train_ratio: Ratio of data to use for training
            random_seed: Random seed for reproducible splits
        """
        self.output_dir = Path(output_dir)
        self.target_num_patches = target_num_patches
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.features_dir = self.output_dir / "features"
        self.train_images_dir = self.images_dir / "train"
        self.val_images_dir = self.images_dir / "val"
        self.train_features_dir = self.features_dir / "train"
        self.val_features_dir = self.features_dir / "val"
        
        # Create all directories
        for dir_path in [self.train_images_dir, self.val_images_dir, 
                        self.train_features_dir, self.val_features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Created output directories in {self.output_dir}")

    def download_and_load_dataset(self, repo_id: str, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Download dataset from HuggingFace and convert to list of records.
        
        Args:
            repo_id: HuggingFace dataset repository ID
            token: HF token for private datasets
            
        Returns:
            List of dataset records
        """
        print(f"📥 Downloading dataset from {repo_id}...")
        
        try:
            dataset = load_dataset(repo_id, token=token)
            records = []
            
            # Handle different dataset splits
            if 'train' in dataset:
                records.extend(list(dataset['train']))
            elif 'test' in dataset:
                records.extend(list(dataset['test']))
            else:
                # If no splits, assume it's a single dataset
                records.extend(list(dataset))
                
            print(f"✓ Loaded {len(records)} records from dataset")
            return records
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise

    def save_image_from_pil(self, pil_image: Image.Image, save_path: Path) -> bool:
        """
        Save PIL Image to local path.
        
        Args:
            pil_image: PIL Image object
            save_path: Path to save image
            
        Returns:
            True if successful, False otherwise
        """
        from olmo.io import file_open, file_exists
        if file_exists(save_path):
            print(f"✓ Image already exists at {save_path}, skipping...")
            return True
        try:
            import io
            with file_open(save_path, 'wb') as f:
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                f.write(img_byte_arr.getvalue())
            return True
        except Exception as e:
            print(f"❌ Failed to save image to {save_path}: {e}")
            return False

    def extract_and_save_features(self, image_path: Path, features_save_path: Path) -> bool:
        """
        Extract SigLIP features from image and save to disk.
        
        Args:
            image_path: Path to image file
            features_save_path: Path to save features
            
        Returns:
            True if successful, False otherwise
        """
        from olmo.io import file_open, file_exists
        if file_exists(features_save_path):
            print(f"✓ Feature already exists at {features_save_path}, skipping...")
            return True
        try:
            # Extract features using the existing function
            features = get_interpolated_patch_features(
                str(image_path), 
                self.target_num_patches
            )
            
            # Save features as numpy array
            with file_open(features_save_path, 'wb') as f:
                np.save(f, features.cpu().numpy())
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract features from {image_path}: {e}")
            return False

    def process_single_record(self, record: Dict[str, Any], record_id: int, split: str) -> Dict[str, Any]:
        """
        Process a single record: save images, extract features, update paths.
        
        Args:
            record: Dataset record
            record_id: Unique record identifier
            split: 'train' or 'val'
            
        Returns:
            Updated record with local paths
        """
        processed_record = record.copy()
        
        # Determine output directories based on split
        if split == 'train':
            images_dir = self.train_images_dir
            features_dir = self.train_features_dir
        else:
            images_dir = self.val_images_dir
            features_dir = self.val_features_dir
        
        # Process main image if it exists
        if 'image' in record and record['image'] is not None:
            image_filename = f"record_{record_id}_main.png"
            image_path = images_dir / image_filename
            
            if self.save_image_from_pil(record['image'], image_path):
                processed_record['image_path'] = str(image_path)
                
            #     # Extract features for main image
            #     features_filename = f"record_{record_id}_main_features.npy"
            #     features_path = features_dir / features_filename
                
            #     if self.extract_and_save_features(image_path, features_path):
            #         processed_record['image_features_path'] = str(features_path)

        # Process image_output if it exists (only for train split)
        if split == 'train' and 'image_output' in record and record['image_output'] is not None:
            output_image_filename = f"record_{record_id}_output.png"
            output_image_path = images_dir / output_image_filename
            
            if self.save_image_from_pil(record['image_output'], output_image_path):
                processed_record['image_output_path'] = str(output_image_path)
                
                # Extract features for output image
                output_features_filename = f"record_{record_id}_output_features.npy"
                output_features_path = features_dir / output_features_filename
                
                if self.extract_and_save_features(output_image_path, output_features_path):
                    processed_record['image_output_features_path'] = str(output_features_path)
        
        # Remove PIL Image objects to make record JSON serializable
        if 'image' in processed_record:
            del processed_record['image']
        if 'image_output' in processed_record:
            del processed_record['image_output']
            
        return processed_record

    def split_dataset(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into train and validation sets.
        
        Args:
            records: List of dataset records
            
        Returns:
            Tuple of (train_records, val_records)
        """
        print(f"🔀 Splitting dataset into train ({self.train_ratio:.1%}) and val ({1-self.train_ratio:.1%})...")
        
        train_records, val_records = train_test_split(
            records, 
            train_size=self.train_ratio,
            random_state=self.random_seed,
            shuffle=True
        )
        
        print(f"✓ Split: {len(train_records)} train, {len(val_records)} val records")
        return train_records, val_records

    def process_records(self, records: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        """
        Process all records for a given split.
        
        Args:
            records: List of records to process
            split: 'train' or 'val'
            
        Returns:
            List of processed records
        """
        print(f"🔄 Processing {len(records)} {split} records...")
        
        processed_records = []
        
        for i, record in enumerate(tqdm(records, desc=f"Processing {split}")):
            try:
                processed_record = self.process_single_record(record, i, split)
                processed_records.append(processed_record)
            except Exception as e:
                print(f"❌ Failed to process {split} record {i}: {e}")
                continue
        
        print(f"✓ Successfully processed {len(processed_records)}/{len(records)} {split} records")
        return processed_records

    def save_json_dataset(self, records: List[Dict[str, Any]], filename: str):
        """
        Save records to JSON file.
        
        Args:
            records: List of records to save
            filename: Output filename
        """
        output_path = self.output_dir / filename

        save_to = []
        keys_to_exclude = [
            'image_path',
            'question',
            'label',
            'image_output_features_path'
        ]

        for idx, r in enumerate(records):
            id_ = idx
            # imagename is the original image path
            imgname = r['image_path'].replace("/datadrive_a/linjie/blob/vigstandard_data/linjli/debug_output/UW/mahtab/Umolmo/Data", "./Data")
            query = r["question"].replace("<image>", "")
            label = r["label"]
            image_output_paths = [
                r["image_output_features_path"].replace("/datadrive_a/linjie/blob/vigstandard_data/linjli/debug_output/UW/mahtab/Umolmo/Data", "./Data")
                ] if "image_output_features_path" in r else []
            others = {}
            for key in r.keys():
                if key in keys_to_exclude:
                    continue
                others[key] = r[key]

            save_to.append({
                "id": id_,
                "imgname": imgname,
                "query": query,
                "label": label,
                "image_output_paths": image_output_paths,
                **others
            })

        try:
            from olmo.io import file_open
            with file_open(output_path, 'w') as f:
                json.dump(save_to, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(save_to)} records to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save {filename}: {e}")

    def create_val_without_outputs(self, val_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create validation records without image outputs (for inference).
        
        Args:
            val_records: Original validation records
            
        Returns:
            Validation records without image outputs
        """
        val_no_outputs = []
        
        for record in val_records:
            record_copy = record.copy()
            
            # Remove image output related fields
            fields_to_remove = [
                'image_output_path', 
                'image_output_features_path',
                'image_output_paths'  # if this field exists
            ]
            
            for field in fields_to_remove:
                if field in record_copy:
                    del record_copy[field]
            
            val_no_outputs.append(record_copy)
        
        return val_no_outputs

    def run_full_pipeline(self, repo_id: str, token: Optional[str] = None):
        """
        Run the complete pipeline: download, process, split, and save.
        
        Args:
            repo_id: HuggingFace dataset repository ID
            token: HF token for private datasets
        """
        print("🚀 Starting SigLIP feature extraction pipeline...")
        
        # Step 1: Download dataset
        records = self.download_and_load_dataset(repo_id, token)
        
        # Step 2: Split dataset
        train_records, val_records = self.split_dataset(records)
        
        # Step 3: Process train records (with image outputs)
        processed_train = self.process_records(train_records, 'train')
        
        # Step 4: Process val records (with image outputs for features)
        processed_val = self.process_records(val_records, 'val')
        
        # Step 5: Create val without outputs
        val_no_outputs = self.create_val_without_outputs(processed_val)
        
        # Step 6: Save JSON files
        self.save_json_dataset(processed_train, 'train.json')
        self.save_json_dataset(val_no_outputs, 'val.json')
        self.save_json_dataset(processed_val, 'val_with_outputs.json')  # Keep for reference
        
        # Step 7: Print summary
        self.print_summary(processed_train, processed_val, val_no_outputs)

    def print_summary(self, train_records: List[Dict[str, Any]], 
                     val_records: List[Dict[str, Any]], 
                     val_no_outputs: List[Dict[str, Any]]):
        """Print pipeline completion summary."""
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Train records: {len(train_records)} (with image outputs)")
        print(f"📊 Val records: {len(val_no_outputs)} (without image outputs)")
        print(f"📊 Val with outputs: {len(val_records)} (for reference)")
        print(f"🖼️  Target patches per image: {self.target_num_patches}")
        print("\n📁 Generated files:")
        print(f"  • train.json - Training data with image outputs")
        print(f"  • val.json - Validation data without image outputs")
        print(f"  • val_with_outputs.json - Validation data with outputs (reference)")
        print(f"  • images/train/ - Training images")
        print(f"  • images/val/ - Validation images")
        print(f"  • features/train/ - Training SigLIP features")
        print(f"  • features/val/ - Validation SigLIP features")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract SigLIP features and split HF dataset")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="HuggingFace dataset repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--target_patches", type=int, default=576,
                       help="Number of patches to interpolate SigLIP features to")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training (0.0-1.0)")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace token for private datasets")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SigLIPFeatureExtractor(
        output_dir=args.output_dir,
        target_num_patches=args.target_patches,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )
    
    # Run pipeline
    extractor.run_full_pipeline(args.repo_id, args.token)


# Example usage function
def example_usage():
    """Example of how to use the feature extractor."""
    
    # Configuration
    repo_id = "linjieli222/frozen-lake-action-safe-single-image-tag-20k-cot"  # Replace with actual repo
    output_dir = "/datadrive_a/linjie/blob/vigstandard_data/linjli/debug_output/UW/mahtab/Umolmo/Data/torch_datasets/frozen-lake-action-safe-single-image-tag-20k-cot"
    target_patches = 64  # 24x24 grid
    train_ratio = 0.8
    token = None  # Set if using private dataset
    
    # Initialize and run
    extractor = SigLIPFeatureExtractor(
        output_dir=output_dir,
        target_num_patches=target_patches,
        train_ratio=train_ratio
    )
    token = os.environ.get("HF_TOKEN")
    extractor.run_full_pipeline(repo_id, token)


if __name__ == "__main__":
    example_usage()
