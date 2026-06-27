#!/usr/bin/env python3
"""Public release module documentation."""
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scoring import load_qalign_model, compute_qalign_score


def main():
    parser = argparse.ArgumentParser(description="Score generated visual_refinement images with Q-Align")
    parser.add_argument("--generated-dir", type=str, required=True,
                       help="Directory containing generated images")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to benchmark JSONL file")
    parser.add_argument("--degraded-dir", type=str, required=True,
                       help="Base directory for degraded (low-quality) images")
    parser.add_argument("--gt-dir", type=str, required=True,
                       help="Base directory for GT (high-quality) images")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save scoring results")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for Q-Align model")
    parser.add_argument("--shots", nargs='+', default=None,
                       help="List of shots to evaluate (e.g., 0 1 5). If provided, --generated-dir and --output-path can contain {shot} placeholder.")

    args = parser.parse_args()


    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return


    if not os.path.exists(args.degraded_dir):
        print(f"Error: Degraded image directory not found: {args.degraded_dir}")
        return

    if not os.path.exists(args.gt_dir):
        print(f"Error: GT image directory not found: {args.gt_dir}")
        return


    print(f"Loading Q-Align model on {args.device}...")
    qalign_model = load_qalign_model(device=args.device)
    print("Q-Align model loaded!")


    print(f"\nLoading benchmark data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} samples")


    if args.shots:
        shots_list = args.shots
        print(f"Running evaluation for shots: {shots_list}")
    else:
        shots_list = [None]

    for shot in shots_list:

        if shot is not None:
            try:
                current_generated_dir = args.generated_dir.format(shot=shot)
                current_output_path = args.output_path.format(shot=shot)
            except KeyError:
                current_generated_dir = args.generated_dir
                current_output_path = args.output_path

            print(f"\n=== Processing Shot: {shot} ===")
        else:
            current_generated_dir = args.generated_dir
            current_output_path = args.output_path


        if not os.path.exists(current_generated_dir):
            print(f"Error: Generated directory not found: {current_generated_dir}")
            continue


        print(f"Scoring generated images from {current_generated_dir}...")
        results = []
        total_efficiency = 0.0
        valid_count = 0
        missing_count = 0

        for item in tqdm(data, desc=f"Scoring {shot if shot else ''}"):
            sample_id = item.get('sample_id', item.get('id', 'unknown'))


            gen_img_path = os.path.join(current_generated_dir, f"perfected_{sample_id}.png")

            if not os.path.exists(gen_img_path):
                missing_count += 1
                results.append({
                    'sample_id': sample_id,
                    'generated_path': gen_img_path,
                    'qalign_score_input': item.get('score_l'),
                    'qalign_score_output': None,
                    'qalign_score_output_quality': None,
                    'qalign_score_output_aesthetics': None,
                    'qalign_score_gt': item.get('score_h'),
                    'efficiency': None,
                    'error': 'Generated image not found'
                })
                continue


            s_in = item.get('score_l')
            s_GT = item.get('score_h')


            try:
                score_dict = compute_qalign_score(gen_img_path, qalign_model)
                if score_dict:
                    s_out = score_dict['total_score']
                    s_out_quality = score_dict['quality_score']
                    s_out_aesthetics = score_dict['aesthetics_score']
                else:
                    s_out = None
                    s_out_quality = None
                    s_out_aesthetics = None
            except Exception as e:
                print(f"  Error scoring {sample_id}: {e}")
                s_out = None
                s_out_quality = None
                s_out_aesthetics = None


            # Efficiency = (s_out - s_in) / (5 - s_in) * 100
            efficiency = None
            if s_in is not None and s_out is not None:
                denominator = 5.0 - s_in
                if abs(denominator) > 1e-6:
                    efficiency = (s_out - s_in) / denominator * 100
                    total_efficiency += efficiency
                    valid_count += 1

            results.append({
                'sample_id': sample_id,
                'generated_path': gen_img_path,
                'qalign_score_input': s_in,
                'qalign_score_output': s_out,
                'qalign_score_output_quality': s_out_quality,
                'qalign_score_output_aesthetics': s_out_aesthetics,
                'qalign_score_gt': s_GT,
                'efficiency': efficiency
            })


        mean_efficiency = total_efficiency / valid_count if valid_count > 0 else 0.0

        print(f"\n=== Scoring Results ({'Shot: ' + str(shot) if shot else 'Single Run'}) ===")
        print(f"Total samples: {len(data)}")
        print(f"Valid scores: {valid_count}")
        print(f"Missing images: {missing_count}")
        print(f"Mean Efficiency: {mean_efficiency:.2f}%")


        result_data = {
            'shot': shot,
            'mean_efficiency': mean_efficiency,
            'valid_count': valid_count,
            'total_count': len(data),
            'missing_count': missing_count,
            'results': results
        }

        os.makedirs(os.path.dirname(current_output_path), exist_ok=True)
        with open(current_output_path, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {current_output_path}")


if __name__ == "__main__":
    main()
