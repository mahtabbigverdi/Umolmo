#!/usr/bin/env python3
"""
Evaluation script for FrozenLake planning task predictions.

This script processes prediction files in JSON format from model shards,
extracts path predictions from boxed answers, and evaluates them using
manual FrozenLake simulation for task success rate calculation.

Usage:
    python evaluate_frozenlake_plan.py --prediction_dir /path/to/predictions/
    python evaluate_frozenlake_plan.py --prediction_dir /path/to/predictions/
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


def extract_boxed_path(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{} entry from the given text.
    
    Args:
        text (str): Text containing boxed paths
        
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


def parse_action_sequence(path_str: str) -> List[str]:
    """
    Parse action sequence from the path string.
    
    Args:
        path_str (str): Path string from prediction
        
    Returns:
        List[str]: List of valid actions ['L', 'R', 'U', 'D']
    """
    if not path_str:
        return []
    
    # Preprocess the path string
    path_str = path_str.lower().strip()
    actions = []
    
    # Split by comma and extract actions
    for action in path_str.split(','):
        action = action.strip().upper()
        if action in ['L', 'R', 'U', 'D']:
            actions.append(action)
    
    return actions


def load_ground_truth_data(gt_file: str) -> Dict[str, Dict]:
    """
    Load ground truth data containing maze_map for each example.
    
    Args:
        prediction_dir (str): Directory containing prediction files
        
    Returns:
        Dict[str, Dict]: Ground truth data indexed by example_id
    """
    gt_file = Path(gt_file)

    # Look for ground truth JSON file in the prediction directory
    # This should contain the maze_map for each example
    
    if not gt_file.exists():
        # Try alternative names

        raise FileNotFoundError(
            f"Ground truth file {gt_file} not found. "
        )
    
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Convert to dict indexed by example_id if it's a list
        if isinstance(gt_data, list):
            gt_dict = {}
            for item in gt_data:
                if 'example_id' in item:
                    gt_dict[str(item['example_id'])] = item
                elif 'id' in item:
                    gt_dict[str(item['id'])] = item
            return gt_dict
        else:
            return gt_data
            
    except Exception as e:
        logger.error(f"Error loading ground truth data: {e}")
        return {}


def manual_frozenlake_simulation(actions: List[str],
                                 map_text: List[str]) -> tuple:
    """
    Manual simulation of FrozenLake environment.
    
    Args:
        actions (List[str]): List of actions ['L', 'R', 'U', 'D']
        map_text (List[str]): Map description as list of strings
        
    Returns:
        tuple: (success, error_message)
    """
    if not actions or not map_text:
        return 0, "Invalid actions or map"
    
    # Find start position
    start_pos = None
    goal_pos = None
    rows, cols = len(map_text), len(map_text[0])
    
    for i, row in enumerate(map_text):
        for j, cell in enumerate(row):
            if cell == 'S':
                start_pos = (i, j)
            elif cell == 'G':
                goal_pos = (i, j)
    
    if start_pos is None:
        return 0, "No start position found"
    if goal_pos is None:
        return 0, "No goal position found"
    
    # Simulate movement
    current_pos = start_pos
    action_mapping = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    for action in actions:
        if action not in action_mapping:
            continue
            
        # Calculate new position
        dr, dc = action_mapping[action]
        new_row = current_pos[0] + dr
        new_col = current_pos[1] + dc
        
        # Check bounds
        if 0 <= new_row < rows and 0 <= new_col < cols:
            current_pos = (new_row, new_col)
            cell = map_text[new_row][new_col]
            
            # Check if fell into hole
            if cell == 'H':
                return 0, "Path leads to hole"
            # Check if reached goal
            elif cell == 'G':
                return 1, "Path leads to goal"
        else:
            # Out of bounds
            return 0, "Path goes out of bounds"
    
    # Didn't reach goal
    if current_pos == goal_pos:
        return 1, "Path leads to goal"
    else:
        return 0, "Path does not reach goal"


def evaluate_planning(prediction: str, metadata: Dict) -> tuple:
    """
    Evaluate path planning task using manual FrozenLake simulation.
    
    Args:
        prediction (str): Predicted action sequence
        metadata (Dict): Metadata containing maze_map
        
    Returns:
        tuple: (success_rate, error_message)
    """
    if prediction is None:
        return 0, "No prediction"
    
    # Parse action sequence
    actions = parse_action_sequence(prediction)
    
    if not actions:
        return 0, "No valid actions found"

    # Get maze map from metadata
    if "maze_map" not in metadata:
        return 0, "No maze_map in metadata"

    rows = metadata["maze_map"]

    # Use manual simulation
    return manual_frozenlake_simulation(actions, rows)


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


def evaluate_predictions(predictions: List[Dict],
                         ground_truth: Dict[str, Dict]) -> Dict:
    """
    Evaluate planning predictions.
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries
        ground_truth (Dict[str, Dict]): Ground truth data by example_id
        
    Returns:
        Dict: Evaluation results
    """
    successful = 0
    total = 0
    no_prediction_path = 0
    no_ground_truth = 0
    extraction_errors = 0
    validation_errors = 0
    
    detailed_results = []
    
    for i, pred_item in enumerate(predictions):
        try:
            # Extract prediction and ground truth
            prediction_text = pred_item.get("prediction", "")
            example_id = str(pred_item.get("example_id", i))
            
            # Extract path from boxed answer
            predicted_path = extract_boxed_path(prediction_text)
            
            # Track extraction success
            if predicted_path is None:
                no_prediction_path += 1
                continue
            
            # Get ground truth metadata
            if example_id not in ground_truth:
                no_ground_truth += 1
                continue
            
            gt_metadata = ground_truth[example_id]
            
            # Evaluate the path
            if "maze_map" in gt_metadata:
                success, error_msg = evaluate_planning(
                    predicted_path, gt_metadata
                )
                total += 1
                if success:
                    successful += 1
            else:
                raise ValueError("No maze_map in ground truth metadata")
            
            # Store detailed result
            pred_text_short = (
                prediction_text[:200] + "..."
                if len(prediction_text) > 200
                else prediction_text
            )
            
            detailed_results.append({
                "example_id": example_id,
                "predicted_path": predicted_path,
                "parsed_actions": parse_action_sequence(predicted_path),
                "success": bool(success),
                "error_message": error_msg,
                "prediction_text": pred_text_short,
                "has_ground_truth": example_id in ground_truth,
                "maze_map": gt_metadata.get("maze_map") if example_id in ground_truth else False
            })
            
        except Exception as e:
            logger.error(f"Error processing prediction {i}: {e}")
            extraction_errors += 1
            continue
    
    # Calculate success rate
    success_rate = successful / total if total > 0 else 0
    
    results = {
        "success_rate": success_rate,
        "successful": successful,
        "total": total,
        "total_predictions": len(predictions),
        "no_prediction_path": no_prediction_path,
        "no_ground_truth": no_ground_truth,
        "extraction_errors": extraction_errors,
        "validation_errors": validation_errors,
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
    print("PLANNING EVALUATION SUMMARY")
    print("="*60)
    print(f"Total predictions loaded: {results['total_predictions']}")
    
    extractable_paths = (
        results['total_predictions'] - results['no_prediction_path']
    )
    print(f"Predictions with extractable paths: {extractable_paths}")
    
    available_gt = (
        results['total_predictions'] - results['no_ground_truth']
    )
    print(f"Predictions with ground truth: {available_gt}")
    
    print(f"Successfully evaluated: {results['total']}")
    print(f"Extraction errors: {results['extraction_errors']}")
    print()
    print(f"Successful paths: {results['successful']}")
    print(f"Total evaluated: {results['total']}")
    
    success_rate_pct = results['success_rate']*100
    print(f"SUCCESS RATE: {results['success_rate']:.4f} "
          f"({success_rate_pct:.2f}%)")
    print("="*60)
    
    if results['no_prediction_path'] > 0:
        print(
            f"Warning: {results['no_prediction_path']} predictions "
            "had no extractable path"
        )
    if results['no_ground_truth'] > 0:
        print(
            f"Warning: {results['no_ground_truth']} predictions "
            "had no ground truth"
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
        description="Evaluate FrozenLake planning predictions"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory containing shard prediction files "
             "(shard0predictions.json, etc.)"
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Ground truth JSON file"
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
        help="Show examples of failed paths"
    )
    
    args = parser.parse_args()
    
    try:
        # Load predictions
        logger.info(f"Loading predictions from {args.prediction_dir}")
        predictions = load_prediction_files(args.prediction_dir)
        
        if not predictions:
            logger.error("No predictions loaded")
            return
        
        # Load ground truth data
        logger.info("Loading ground truth data...")
        ground_truth = load_ground_truth_data(args.gt_file)
        
        # Evaluate predictions
        logger.info("Evaluating planning predictions...")
        results = evaluate_predictions(predictions, ground_truth)
        
        # Print summary
        print_summary(results)
        
        # Show examples
        if args.show_examples > 0 and results['detailed_results']:
            print(f"\nFirst {args.show_examples} evaluation examples:")
            print("-" * 60)
            examples_to_show = results['detailed_results'][:args.show_examples]
            for i, example in enumerate(examples_to_show):
                print(f"Example {i+1} (ID: {example['example_id']}):")
                print(f"  Predicted Path: '{example['predicted_path']}'")
                print(f"  Parsed Actions: {example['parsed_actions']}")
                print(f"  Maze Map: {example['maze_map']}")
                print(f"  Success: {example['success']}")
                print(f"  Error: {example['error_message']}")
                print(f"  Has GT: {example['has_ground_truth']}")
                print()
        
        # Show error examples
        if args.show_errors and results['detailed_results']:
            failed_examples = [
                r for r in results['detailed_results']
                if not r['success']
            ]
            if failed_examples:
                num_to_show = min(args.show_examples, len(failed_examples))
                print(f"\nFirst {num_to_show} failed examples:")
                print("-" * 60)
                for i, example in enumerate(failed_examples[:num_to_show]):
                    print(
                        f"Failed Example {i+1} "
                        f"(ID: {example['example_id']}):"
                    )
                    print(f"  Predicted Path: '{example['predicted_path']}'")
                    print(f"  Parsed Actions: {example['parsed_actions']}")
                    print(f"  Maze Map: {example['maze_map']}")
                    print(f"  Error: {example['error_message']}")
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
