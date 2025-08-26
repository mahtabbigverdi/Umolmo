#!/usr/bin/env python3
"""
Evaluation script for calculating accuracy from model prediction files.

This script processes prediction files in JSON format from model shards,
extracts boxed answers, and calculates accuracy by comparing predicted
answers with ground truth answers.

Usage:
    python evaluate_boxed_predictions.py --prediction_dir /path/to/predictions/
    python evaluate_boxed_predictions.py --prediction_dir /path/to/predictions/
        --output_file results.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{} entry from the given text.
    
    Args:
        text (str): Text containing boxed answers
        
    Returns:
        Optional[str]: The content inside the last \\boxed{}
                       or None if not found
    """
    if not text:
        return None
    
    # Find all boxed patterns in the text
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text)
    
    if matches:
        # Return the last match (most recent boxed answer)
        return matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    Args:
        answer (str): Raw answer text
        
    Returns:
        str: Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()
    
    # Remove common punctuation that doesn't affect meaning
    answer = re.sub(r'[.,;:!?]$', '', answer)
    
    # Handle common variations
    answer = answer.replace("yes", "yes").replace("no", "no")
    
    return answer


def load_prediction_files(prediction_dir: str) -> List[Dict]:
    """
    Load all prediction files from the given directory.
    
    Args:
        prediction_dir (str): Directory containing prediction files
        
    Returns:
        List[Dict]: Combined predictions from all shards
    """
    prediction_dir = Path(prediction_dir)
    
    if not prediction_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}"
        )
    
    all_predictions = []
    shard_files = []
    
    # Look for shard prediction files
    for file_path in prediction_dir.glob("shard*predictions.json"):
        shard_files.append(file_path)
    
    if not shard_files:
        raise FileNotFoundError(
            f"No shard prediction files found in {prediction_dir}"
        )
    
    # Sort files to ensure consistent order
    shard_files.sort()
    
    logger.info(
        f"Found {len(shard_files)} shard files: "
        f"{[f.name for f in shard_files]}"
    )
    
    for shard_file in shard_files:
        logger.info(f"Loading predictions from {shard_file.name}")
        
        try:
            with open(shard_file, 'r', encoding='utf-8') as f:
                shard_predictions = json.load(f)
            
            if not isinstance(shard_predictions, list):
                logger.error(
                    f"Expected list in {shard_file.name}, "
                    f"got {type(shard_predictions)}"
                )
                continue
                
            all_predictions.extend(shard_predictions)
            logger.info(
                f"Loaded {len(shard_predictions)} predictions "
                f"from {shard_file.name}"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in {shard_file.name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error loading {shard_file.name}: {e}")
            continue
    
    logger.info(f"Total predictions loaded: {len(all_predictions)}")
    return all_predictions


def evaluate_predictions(predictions: List[Dict]) -> Dict:
    """
    Evaluate predictions by comparing extracted boxed answers.
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries
        
    Returns:
        Dict: Evaluation results
    """
    correct = 0
    total = 0
    no_prediction_answer = 0
    no_ground_truth_answer = 0
    extraction_errors = 0
    
    detailed_results = []
    
    for i, pred_item in enumerate(predictions):
        try:
            # Extract prediction and ground truth
            prediction_text = pred_item.get("prediction", "")
            answers_text = pred_item.get("answers", "")
            example_id = pred_item.get("example_id", i)
            
            # Extract boxed answers
            predicted_answer = extract_boxed_answer(prediction_text)
            ground_truth_answer = extract_boxed_answer(answers_text)
            
            # Track extraction success
            if predicted_answer is None:
                no_prediction_answer += 1
            if ground_truth_answer is None:
                no_ground_truth_answer += 1
            
            # Only evaluate if both answers were extracted
            if (predicted_answer is not None and
                    ground_truth_answer is not None):
                # Normalize answers for comparison
                pred_normalized = normalize_answer(predicted_answer)
                gt_normalized = normalize_answer(ground_truth_answer)
                
                is_correct = pred_normalized == gt_normalized
                if is_correct:
                    correct += 1
                total += 1
                
                # Store detailed result
                pred_text_short = (
                    prediction_text[:200] + "..."
                    if len(prediction_text) > 200
                    else prediction_text
                )
                answers_text_short = (
                    answers_text[:200] + "..."
                    if len(answers_text) > 200
                    else answers_text
                )
                
                detailed_results.append({
                    "example_id": example_id,
                    "predicted_answer": predicted_answer,
                    "ground_truth_answer": ground_truth_answer,
                    "predicted_normalized": pred_normalized,
                    "ground_truth_normalized": gt_normalized,
                    "is_correct": is_correct,
                    "prediction_text": pred_text_short,
                    "answers_text": answers_text_short
                })
            
        except Exception as e:
            logger.error(f"Error processing prediction {i}: {e}")
            extraction_errors += 1
            continue
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_predictions": len(predictions),
        "no_prediction_answer": no_prediction_answer,
        "no_ground_truth_answer": no_ground_truth_answer,
        "extraction_errors": extraction_errors,
        "detailed_results": detailed_results
    }
    
    return results


def print_summary(results: Dict):
    """
    Print evaluation summary.
    
    Args:
        results (Dict): Evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total predictions loaded: {results['total_predictions']}")
    
    extractable_preds = (
        results['total_predictions'] - results['no_prediction_answer']
    )
    print(f"Predictions with extractable answers: {extractable_preds}")
    
    extractable_gt = (
        results['total_predictions'] - results['no_ground_truth_answer']
    )
    print(f"Ground truth with extractable answers: {extractable_gt}")
    
    print(f"Successfully evaluated: {results['total']}")
    print(f"Extraction errors: {results['extraction_errors']}")
    print()
    print(f"Correct predictions: {results['correct']}")
    print(f"Total evaluated: {results['total']}")
    
    accuracy_pct = results['accuracy']*100
    print(f"ACCURACY: {results['accuracy']:.4f} ({accuracy_pct:.2f}%)")
    print("="*60)
    
    if results['no_prediction_answer'] > 0:
        print(
            f"Warning: {results['no_prediction_answer']} predictions "
            "had no extractable boxed answer"
        )
    if results['no_ground_truth_answer'] > 0:
        print(
            f"Warning: {results['no_ground_truth_answer']} ground truth "
            "had no extractable boxed answer"
        )
    if results['extraction_errors'] > 0:
        print(
            f"Warning: {results['extraction_errors']} predictions "
            "had extraction errors"
        )


def save_results(results: Dict, output_file: str):
    """
    Save results to JSON file.
    
    Args:
        results (Dict): Evaluation results
        output_file (str): Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions with boxed answers"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory containing shard prediction files "
             "(shard0predictions.json, etc.)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for detailed results (optional)"
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Number of example results to show (default: 5)"
    )
    parser.add_argument(
        "--show_errors",
        action="store_true",
        help="Show examples of incorrect predictions"
    )

    args = parser.parse_args()

    try:
        # Load predictions
        logger.info(f"Loading predictions from {args.prediction_dir}")
        predictions = load_prediction_files(args.prediction_dir)

        if not predictions:
            logger.error("No predictions loaded")
            return

        # Evaluate predictions
        logger.info("Evaluating predictions...")
        results = evaluate_predictions(predictions)

        # Print summary
        print_summary(results)

        # Show examples
        if args.show_examples > 0 and results['detailed_results']:
            print(f"\nFirst {args.show_examples} evaluation examples:")
            print("-" * 60)
            examples_to_show = results['detailed_results'][:args.show_examples]
            for i, example in enumerate(examples_to_show):
                print(f"Example {i+1} (ID: {example['example_id']}):")
                print(f"  Predicted: '{example['predicted_answer']}'")
                print(f"  Ground Truth: '{example['ground_truth_answer']}'")
                print(f"  Correct: {example['is_correct']}")
                print()

        # Show error examples
        if args.show_errors and results['detailed_results']:
            incorrect_examples = [
                r for r in results['detailed_results']
                if not r['is_correct']
            ]
            if incorrect_examples:
                num_to_show = min(args.show_examples, len(incorrect_examples))
                print(f"\nFirst {num_to_show} incorrect examples:")
                print("-" * 60)
                for i, example in enumerate(incorrect_examples[:num_to_show]):
                    print(
                        f"Incorrect Example {i+1} "
                        f"(ID: {example['example_id']}):"
                    )
                    print(f"  Predicted: '{example['predicted_answer']}'")
                    print(
                        f"  Ground Truth: "
                        f"'{example['ground_truth_answer']}'"
                    )
                    print(f"  Prediction text: {example['prediction_text']}")
                    print()

        # Save results if requested
        if args.output_file:
            save_results(results, args.output_file)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
