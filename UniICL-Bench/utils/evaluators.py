"""
Model-Agnostic Evaluation Functions
===================================

All evaluation functions for different benchmark tasks.
These functions are model-agnostic and work with any inferencer
that implements the interleave_inference() method.

Each evaluation function follows the pattern:
    def eval_<task>(inferencer, data_path, image_dir, output_path, ...)

The inferencer must implement:
    - interleave_inference(input_lists, understanding_output, ...)
      Returns a list where the last element is the model's text output
"""

import json
import os
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from bert_score import score as bert_score
import torch

# Import from utils modules
from .bbox import parse_bbox, normalize_bbox, compute_iou
from .icl import build_icl_input
from .judge import call_vllm_judge, mllm_assisted_extraction
from .parsing import (
    extract_answer_from_tags,
    extract_option_letter,
    extract_action_from_tags,
    parse_mcq_options,
    get_instruction_text,
    extract_score_from_annotation,
    extract_label_from_annotation,
    parse_option_label,
)
from .scoring import compute_qalign_score, compute_hpsv3_score, compute_clip_score, load_clip_model, load_hpsv3_model

def eval_grounding(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Visual Grounding ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    total_iou = 0.0
    valid_count = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Visual Grounding {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")


        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            raise RuntimeError(f"Cannot read image {image_path}. The file may be corrupted or in an unsupported format: {e}")


        demos = item['demos'][:num_demos] if num_demos > 0 else []


        question = item.get('instruction', item.get('text', ''))
        input_list = build_icl_input(demos, image_dir, image_path, question)


        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for {item['image_name']}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        pred_bbox_raw = parse_bbox(prediction)

        pred_bbox = normalize_bbox(pred_bbox_raw, img_width, img_height)


        gt_bbox_raw = item.get('answer', item.get('annotation', []))
        if isinstance(gt_bbox_raw, str):
            gt_bbox = parse_bbox(gt_bbox_raw)
        elif isinstance(gt_bbox_raw, list):
            gt_bbox = gt_bbox_raw
        else:
            gt_bbox = None


        iou = 0.0
        if pred_bbox is not None and gt_bbox is not None:
            iou = compute_iou(pred_bbox, gt_bbox)
            total_iou += iou
            valid_count += 1

        results.append({
            'image_name': item['image_name'],
            'question': item.get('instruction', item.get('text', '')),
            'ground_truth': gt_bbox,
            'prediction': prediction,
            'pred_bbox_raw': pred_bbox_raw,
            'pred_bbox_normalized': pred_bbox,
            'image_size': [img_width, img_height],
            'iou': iou,
            'inference_failed': inference_failed,
            'error_message': error_message
        })


    mean_iou = total_iou / valid_count if valid_count > 0 else 0.0
    print(f"\nMean IoU: {mean_iou:.4f} ({valid_count}/{len(data)} valid predictions)")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['image_name']}: {r['error_message']}")
        print(f"{'='*80}\n")


    result_data = {
        'mean_iou': mean_iou,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_attr_rec_gen(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Attribute Recognition ({num_demos}-shot) ===")
    
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    correct = 0
    total = 0
    
    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )
    
    for item in tqdm(data, desc=f"Attribute Recognition {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        

        demos = item['demos'][:num_demos] if num_demos > 0 else []
        

        input_list = []


        for demo in demos:
            demo_img_path = os.path.join(image_dir, demo['image_name'])


            if not os.path.exists(demo_img_path):
                raise FileNotFoundError(f"Attribute Recognition demo image required but not found: {demo_img_path}")

            demo_img = Image.open(demo_img_path).convert("RGB")
            demo_answer = demo.get('answer', demo.get('annotation', ''))


            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo['text'])
            input_list.append(f"\nAssistant: {demo_answer}")


        target_img = Image.open(image_path).convert("RGB")
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(target_img)
        input_list.append(item['text'])
        input_list.append("\nAssistant:")


        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for {item['image_name']}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        judge_prompt = (
            f"Question: {item.get('instruction', item.get('text', ''))}\n"
            f"Ground Truth: {item.get('answer', item.get('annotation', ''))}\n"
            f"Prediction: {prediction}\n"
            "Compare the prediction with the ground truth strictly. "
            "The prediction must contain all attributes mentioned in the ground truth and should not miss any. "
            "For example, if GT is 'yellow and green', prediction 'yellow' is Wrong. "
            "Is the prediction correct? Respond with only 'Yes' or 'No'."
        )
        
        judge_response = call_vllm_judge(judge_prompt, image_path)
        is_correct = "yes" in judge_response.lower()
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'image_name': item['image_name'],
            'question': item.get('instruction', item.get('text', '')),
            'ground_truth': item.get('answer', item.get('annotation', '')),
            'prediction': prediction,
            'judge_response': judge_response,
            'correct': is_correct,
            'inference_failed': inference_failed,
            'error_message': error_message
        })
    

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['image_name']}: {r['error_message']}")
        print(f"{'='*80}\n")


    result_data = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")
    return results


def eval_vqa_gen(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Scene Reasoning ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    total_score = 0.0
    total_bert_f1 = 0.0
    valid_count = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Scene Reasoning {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        demos = item['demos'][:num_demos] if num_demos > 0 else []


        input_list = []
        for demo in demos:
            demo_img_path = os.path.join(image_dir, demo['image_name'])


            if not os.path.exists(demo_img_path):
                raise FileNotFoundError(f"Scene Reasoning demo image required but not found: {demo_img_path}")

            demo_img = Image.open(demo_img_path).convert("RGB")
            demo_question = demo.get('instruction', demo.get('text', ''))
            demo_answer = demo.get('answer', demo.get('annotation', ''))


            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_question)
            input_list.append(f"\nAssistant: {demo_answer}")

        target_img = Image.open(image_path).convert("RGB")
        question = item.get('instruction', item.get('text', ''))
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(target_img)
        input_list.append(question)
        input_list.append("\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for {item['image_name']}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        bert_precision = 0.0
        bert_recall = 0.0
        bert_f1 = 0.0

        if prediction:
            ground_truth = item.get('answer', item.get('annotation', ''))
            try:
                P, R, F1 = bert_score([prediction], [ground_truth], lang='en', verbose=False)
                bert_precision = P.item()
                bert_recall = R.item()
                bert_f1 = F1.item()
                total_bert_f1 += bert_f1
            except Exception as e:
                print(f"Error computing BertScore: {e}")


        score = 0.0
        rationale = ""

        if prediction:
            judge_prompt = (
                "You are an expert evaluator for visual question answering tasks. "
                "Evaluate the prediction against the ground truth answer with strict, fine-grained criteria.\n\n"
                f"**Question**: {item.get('instruction', item.get('text', ''))}\n"
                f"**Ground Truth Answer**: {item.get('answer', item.get('annotation', ''))}\n"
                f"**Model's Prediction**: {prediction}\n\n"
                "First, provide a detailed rationale (2-3 sentences) analyzing:\n"
                "1) Whether all key facts from ground truth are present\n"
                "2) Whether there are any factual errors or contradictions\n"
                "3) Whether the level of detail is appropriate\n\n"
                "Then provide a score from 1-100 based on these strict criteria:\n\n"
                "**Scoring Guidelines** (Be very strict - only semantically equivalent answers deserve 90+):\n\n"
                "**90-100 (Excellent)**:\n"
                "- 95-100: Perfect match with ALL key information, no omissions, completely accurate\n"
                "- 90-94: Semantically equivalent, all key facts present, only trivial wording differences\n\n"
                "**75-89 (Good)**:\n"
                "- 85-89: Core answer fully correct, one minor detail missing or slightly imprecise\n"
                "- 80-84: Core answer correct, 2-3 minor details missing or imprecise\n"
                "- 75-79: Core answer correct, but lacks some supporting details from ground truth\n\n"
                "**60-74 (Acceptable)**:\n"
                "- 70-74: Main point correct, but misses important secondary information\n"
                "- 65-69: Main point correct, several key details missing or imprecise\n"
                "- 60-64: Gets general idea right, but significant information gaps\n\n"
                "**40-59 (Partially Correct)**:\n"
                "- 50-59: Some correct elements, but major details missing or inaccurate\n"
                "- 45-49: Contains partial truth, but also has notable errors\n"
                "- 40-44: Minimal correct information, mostly incomplete or inaccurate\n\n"
                "**20-39 (Mostly Wrong)**:\n"
                "- 30-39: Misses the main point, contains major factual errors\n"
                "- 20-29: Almost entirely wrong, minimal overlap with ground truth\n\n"
                "**1-19 (Completely Wrong)**:\n"
                "- 10-19: Completely wrong answer or irrelevant response\n"
                "- 1-9: Nonsensical, unintelligible, or refuses to answer\n\n"
                "**Critical Rules**:\n"
                "- Semantic equivalence is acceptable, but ALL key facts must be present\n"
                "- Missing even one key fact should cap the score at 85\n"
                "- Any factual error should cap the score at 70\n"
                "- Vague or ambiguous answers should score below 60\n"
                "- Be strict: when in doubt between two ranges, choose the lower one\n\n"
                "**Output Format**:\n"
                "Rationale: [Your detailed 2-3 sentence assessment]\n"
                "Score: [score 1-100]"
            )

            judge_response = call_vllm_judge(judge_prompt, image_path)

            try:

                rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Score:|$)', judge_response, re.IGNORECASE | re.DOTALL)
                if rationale_match:
                    rationale = rationale_match.group(1).strip()


                score_match = re.search(r'Score[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    score = min(100.0, max(1.0, score))
                    total_score += score
                    valid_count += 1
            except Exception as e:
                print(f"Error parsing score: {e}")

        results.append({
            'image_name': item['image_name'],
            'question': item.get('instruction', item.get('text', '')),
            'ground_truth': item.get('answer', item.get('annotation', '')),
            'prediction': prediction,
            'rationale': rationale,
            'score': score,
            'bert_precision': bert_precision,
            'bert_recall': bert_recall,
            'bert_f1': bert_f1,
            'inference_failed': inference_failed,
            'error_message': error_message
        })

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    mean_bert_f1 = total_bert_f1 / valid_count if valid_count > 0 else 0.0
    print(f"\nMean Scene Reasoning Score: {mean_score:.2f}/100 ({valid_count}/{len(data)} valid)")
    print(f"Mean BertScore F1: {mean_bert_f1:.4f}")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['image_name']}: {r['error_message']}")
        print(f"{'='*80}\n")

    result_data = {
        'mean_score': mean_score,
        'mean_bert_f1': mean_bert_f1,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_caption_styled(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Style-Aware Caption ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    total_score = 0.0
    total_bert_f1 = 0.0
    valid_count = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Style-Aware Caption {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        demos = item['demos'][:num_demos] if num_demos > 0 else []


        input_list = []
        for demo in demos:
            demo_img_path = os.path.join(image_dir, demo['image_name'])


            if not os.path.exists(demo_img_path):
                raise FileNotFoundError(f"Style-Aware Caption demo image required but not found: {demo_img_path}")

            demo_img = Image.open(demo_img_path).convert("RGB")
            demo_instruction = get_instruction_text(demo)
            demo_answer = demo.get('answer', demo.get('annotation', ''))


            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_instruction)
            input_list.append(f"\nAssistant: {demo_answer}")

        target_img = Image.open(image_path).convert("RGB")
        target_instruction = get_instruction_text(item)
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(target_img)
        input_list.append(target_instruction)
        input_list.append("\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for {item['image_name']}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        bert_precision = 0.0
        bert_recall = 0.0
        bert_f1 = 0.0

        if prediction:
            ground_truth = item.get('answer', item.get('annotation', ''))
            try:
                P, R, F1 = bert_score([prediction], [ground_truth], lang='en', verbose=False)
                bert_precision = P.item()
                bert_recall = R.item()
                bert_f1 = F1.item()
                total_bert_f1 += bert_f1
            except Exception as e:
                print(f"Error computing BertScore: {e}")


        gpt_score = 0.0
        style_score = 0.0
        content_score = 0.0

        if prediction:
            judge_prompt = (
                "You are an expert evaluator for image captioning tasks.\n\n"
                f"**Style Instruction**: {item['instruction']}\n"
                f"**Ground Truth Caption**: {item.get('answer', item.get('annotation', ''))}\n"
                f"**Generated Caption**: {prediction}\n\n"
                "Evaluate on TWO dimensions. Score each from 1-100. DO NOT calculate the total score.\n\n"

                "**1. Style Adherence (1-100)** - Compare with Ground Truth's style\n"
                "- 85-100: Same style as GT (tone, length, format), e.g. both cinematic/poetic/concise\n"
                "- 70-84: Similar style to GT, minor tone or length differences\n"
                "- 50-69: Recognizable style attempt, but noticeably different from GT's style\n"
                "- 25-49: Wrong style or significantly longer/shorter than GT\n"
                "- 1-24: Completely different style, ignores instruction\n"
                "**Note**: If generated caption is much longer than GT (e.g. verbose paragraph vs concise phrase), cap Style at 50.\n\n"

                "**2. Content Accuracy (1-100)**\n"
                "- 85-100: All key visual elements correctly described, no hallucination\n"
                "- 70-84: Core content correct, minor details missing or imprecise\n"
                "- 50-69: Main subject correct, but notable errors or omissions\n"
                "- 25-49: Major content errors or hallucinations present\n"
                "- 1-24: Completely wrong or fabricated content\n\n"

                "**Output Format**:\n"
                "Rationale: [Brief assessment]\n"
                "Style_Score: [1-100]\n"
                "Content_Score: [1-100]"
            )

            judge_response = call_vllm_judge(judge_prompt, image_path)
            try:

                style_match = re.search(r'Style[_\s]*Score[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                content_match = re.search(r'Content[_\s]*Score[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)

                if style_match:
                    style_score = float(style_match.group(1))
                    style_score = min(100.0, max(1.0, style_score))

                if content_match:
                    content_score = float(content_match.group(1))
                    content_score = min(100.0, max(1.0, content_score))


                if style_match and content_match:
                    gpt_score = style_score * 0.6 + content_score * 0.4
                    total_score += gpt_score
                    valid_count += 1
            except Exception as e:
                print(f"Error parsing scores: {e}")

        results.append({
            'image_name': item['image_name'],
            'instruction': item['instruction'],
            'ground_truth': item.get('answer', item.get('annotation', '')),
            'prediction': prediction,
            'style_score': style_score,
            'content_score': content_score,
            'gpt_score': gpt_score,
            'bert_precision': bert_precision,
            'bert_recall': bert_recall,
            'bert_f1': bert_f1,
            'judge_response': judge_response if prediction else "",
            'inference_failed': inference_failed,
            'error_message': error_message
        })

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    mean_bert_f1 = total_bert_f1 / valid_count if valid_count > 0 else 0.0
    print(f"\nMean GPT-Score: {mean_score:.4f} ({valid_count}/{len(data)} valid)")
    print(f"Mean BertScore F1: {mean_bert_f1:.4f}")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['image_name']}: {r['error_message']}")
        print(f"{'='*80}\n")

    result_data = {
        'mean_gpt_score': mean_score,
        'mean_bert_f1': mean_bert_f1,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_t2i(inferencer, data_path, image_dir, output_path, num_demos=3, hps_model=None, gen_output_dir=None):
    """Public release documentation."""
    print(f"\n=== Evaluating Instructional Generation ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]


    if gen_output_dir is None:
        gen_output_dir = os.path.join(os.path.dirname(output_path), "t2i_generated")
    os.makedirs(gen_output_dir, exist_ok=True)

    results = []
    total_score = 0.0
    valid_count = 0

    inference_params = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Instructional Generation {num_demos}shot"):
        demos = item.get('demos', [])[:num_demos] if num_demos > 0 else []



        input_list = []
        for demo in demos:
            demo_prompt = demo.get('instruction', demo.get('text', ''))

            demo_img_path = os.path.join(image_dir, demo['image_name'])


            if not os.path.exists(demo_img_path):
                raise FileNotFoundError(f"Instructional Generation demo image required but not found: {demo_img_path}")


            demo_img = Image.open(demo_img_path).convert("RGB")
            input_list.append(f"\nUser: Generate an image: {demo_prompt}\nAssistant: ")
            input_list.append(demo_img)


        prompt = item.get('instruction', item.get('text', ''))
        if len(demos) > 0:
            input_list.append(f"\nUser: Generate an image: {prompt}\nAssistant:")
        else:
            input_list.append(f"User: Generate an image: {prompt}\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=False,
                think=False,
                **inference_params
            )
            generated_img = output_list[-1]


            gen_img_path = os.path.join(gen_output_dir, os.path.basename(item['image_name']))
            if isinstance(generated_img, Image.Image):
                generated_img.save(gen_img_path)
            else:
                gen_img_path = None
        except Exception as e:
            print(f"Error generating {item['image_name']}: {e}")
            gen_img_path = None


        hpsv3_score = -1.0
        if gen_img_path and os.path.exists(gen_img_path) and hps_model:
            hpsv3_score = compute_hpsv3_score(gen_img_path, prompt, hps_model)
            if hpsv3_score >= 0:
                total_score += hpsv3_score
                valid_count += 1

        results.append({
            'image_name': item['image_name'],
            'prompt': prompt,
            'generated_path': gen_img_path,
            'hpsv3_score': hpsv3_score
        })

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    print(f"\nMean HPSv3-Score: {mean_score:.4f} ({valid_count}/{len(data)} valid)")

    result_data = {
        'mean_hpsv3_score': mean_score,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_i2i_editing(inferencer, data_path, image_dir, output_path, num_demos=3, gen_output_dir=None):
    """Public release documentation."""
    print(f"\n=== Evaluating Image Manipulation ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]


    if gen_output_dir is None:
        gen_output_dir = os.path.join(os.path.dirname(output_path), "i2i_generated")
    os.makedirs(gen_output_dir, exist_ok=True)

    results = []
    total_score = 0.0
    valid_count = 0

    inference_params = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Image Manipulation {num_demos}shot"):

        source_img_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(source_img_path):
            raise FileNotFoundError(f"Image Manipulation source image required but not found: {source_img_path}")

        demos = item.get('demos', [])[:num_demos] if num_demos > 0 else []



        input_list = []
        for demo in demos:

            demo_src_path = os.path.join(image_dir, demo['image_name'])


            demo_edited_rel = demo.get('answer', demo.get('annotation', ''))
            if not demo_edited_rel:
                raise ValueError(f"Image Manipulation demo GT path is empty for demo: {demo}")
            demo_edited_path = os.path.join(image_dir, demo_edited_rel)


            if not os.path.exists(demo_src_path):
                raise FileNotFoundError(f"Image Manipulation demo source image required but not found: {demo_src_path}")

            if not os.path.exists(demo_edited_path):
                raise FileNotFoundError(f"Image Manipulation demo GT image required but not found: {demo_edited_path}")


            demo_img = Image.open(demo_src_path).convert("RGB")
            demo_edited_img = Image.open(demo_edited_path).convert("RGB")
            demo_instruction = demo.get('instruction', demo.get('text', ''))

            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_instruction)
            input_list.append("\nAssistant: ")
            input_list.append(demo_edited_img)


        source_img = Image.open(source_img_path).convert("RGB")
        instruction = item.get('instruction', item.get('text', ''))
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(source_img)
        input_list.append(instruction)
        input_list.append("\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=False,
                think=False,
                **inference_params
            )
            generated_img = output_list[-1]


            gen_img_path = os.path.join(gen_output_dir, f"edited_{os.path.basename(item['image_name'])}")
            if isinstance(generated_img, Image.Image):
                generated_img.save(gen_img_path)
            else:
                gen_img_path = None
        except Exception as e:
            print(f"Error editing {item['image_name']}: {e}")
            gen_img_path = None


        edit_accuracy_score = 0.0
        preservation_score = 0.0
        final_score = 0.0
        rationale = ""

        if gen_img_path and os.path.exists(gen_img_path):
            instruction = item.get('instruction', item.get('text', ''))


            gt_img_rel = item.get('answer', item.get('annotation', ''))
            if not gt_img_rel:
                raise ValueError(f"Image Manipulation GT path is empty for item: {item['image_name']}")
            gt_img_path = os.path.join(image_dir, gt_img_rel)


            if not os.path.exists(gt_img_path):
                raise FileNotFoundError(f"Image Manipulation GT image required but not found: {gt_img_path}")


            judge_prompt = (
                "You are an expert image editing evaluator.\n\n"
                "**You will see 3 images in order:**\n"
                "  1. Original Image - The source image before editing\n"
                "  2. Reference Image (GT) - The ground truth edited result (reference)\n"
                "  3. Model Output - The model's edited output\n\n"
                f"**Edit Instruction**: {instruction}\n\n"
                "Evaluate on TWO dimensions. Score each from 1-100. DO NOT calculate the total score.\n\n"

                "**1. Edit Accuracy (1-100)** - How well does the model output follow the instruction?\n"
                "Compare with the reference image to understand the expected result.\n"
                "- 85-100: All instruction details correctly implemented\n"
                "- 70-84: Core instruction followed, minor details missing or imprecise\n"
                "- 50-69: Partially executed, notable instruction elements incomplete\n"
                "- 25-49: Weak execution, significant misunderstanding or missing elements\n"
                "- 1-24: Wrong edit or fails to follow instruction\n\n"

                "**2. Preservation & Quality (1-100)** - Are unedited regions preserved with good quality?\n"
                "- 85-100: Perfect preservation, no unintended changes, high quality\n"
                "- 70-84: Minor unintended changes, good overall quality\n"
                "- 50-69: Noticeable unintended changes, acceptable quality\n"
                "- 25-49: Significant preservation failures or quality degradation\n"
                "- 1-24: Major corruption or widespread unintended changes\n\n"

                "**Output Format**:\n"
                "Rationale: [Brief assessment]\n"
                "Edit_Accuracy: [1-100]\n"
                "Preservation_Quality: [1-100]"
            )


            images_to_judge = [source_img_path, gt_img_path, gen_img_path]

            judge_response = call_vllm_judge(judge_prompt, images_to_judge)
            try:

                rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Edit_Accuracy|$)', judge_response, re.IGNORECASE | re.DOTALL)
                if rationale_match:
                    rationale = rationale_match.group(1).strip()


                accuracy_match = re.search(r'Edit[_\s]*Accuracy[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                preservation_match = re.search(r'Preservation[_\s]*(?:Quality|&\s*Quality)?[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)

                if accuracy_match:
                    edit_accuracy_score = float(accuracy_match.group(1))
                    edit_accuracy_score = min(100.0, max(1.0, edit_accuracy_score))

                if preservation_match:
                    preservation_score = float(preservation_match.group(1))
                    preservation_score = min(100.0, max(1.0, preservation_score))


                if accuracy_match and preservation_match:
                    final_score = edit_accuracy_score * 0.5 + preservation_score * 0.5
                    total_score += final_score
                    valid_count += 1
            except Exception as e:
                print(f"Error parsing scores: {e}")

        results.append({
            'image_name': item['image_name'],
            'instruction': item.get('instruction', item.get('text', '')),
            'num_demos_requested': num_demos,
            'num_demos_valid': len(demos),
            'source_path': source_img_path,
            'generated_path': gen_img_path,
            'rationale': rationale,
            'edit_accuracy_score': edit_accuracy_score,
            'preservation_quality_score': preservation_score,
            'final_score': final_score
        })

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    print(f"\nMean Image Manipulation Score: {mean_score:.2f}/100 ({valid_count}/{len(data)} valid)")

    result_data = {
        'mean_score': mean_score,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_aesthetic_assessment(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Aesthetic Assessment ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    predictions = []
    ground_truths = []
    mllm_assisted_count = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Aesthetic Assessment {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        demos = item.get('demos', [])[:num_demos] if num_demos > 0 else []


        input_list = []
        for demo in demos:
            demo_img_path = os.path.join(image_dir, demo['image_name'])


            if not os.path.exists(demo_img_path):
                raise FileNotFoundError(f"Aesthetic Assessment demo image required but not found: {demo_img_path}")

            demo_img = Image.open(demo_img_path).convert("RGB")
            demo_instruction = get_instruction_text(demo)
            demo_answer = demo.get('answer', demo.get('annotation', ''))


            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_instruction)
            input_list.append(f"\nAssistant: {demo_answer}")

        target_img = Image.open(image_path).convert("RGB")
        target_instruction = get_instruction_text(item)
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(target_img)
        input_list.append(target_instruction)
        input_list.append("\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for {item['image_name']}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        answer_str = extract_answer_from_tags(prediction)


        if answer_str is None:
            answer_str = mllm_assisted_extraction(
                prediction,
                target_instruction,
                "An integer score between 1-10"
            )
            if answer_str:
                mllm_assisted_count += 1


        pred_score = None
        if answer_str:
            score_match = re.search(r'(\d+(?:\.\d+)?)', answer_str)
            if score_match:
                try:
                    pred_score = float(score_match.group(1))
                except ValueError:
                    pass


        gt_score = extract_score_from_annotation(item.get('answer', item.get('annotation', '')))

        if pred_score is not None and gt_score is not None:
            predictions.append(pred_score)
            ground_truths.append(gt_score)

        results.append({
            'image_name': item['image_name'],
            'question': get_instruction_text(item),
            'ground_truth': gt_score,
            'prediction': prediction,
            'extracted_answer': answer_str,
            'pred_score': pred_score,
            'mllm_assisted': answer_str is not None and extract_answer_from_tags(prediction) is None,
            'inference_failed': inference_failed,
            'error_message': error_message
        })


    srcc = 0.0
    plcc = 0.0
    if len(predictions) >= 2:
        srcc, _ = spearmanr(ground_truths, predictions)
        plcc, _ = pearsonr(ground_truths, predictions)
        srcc = float(srcc) if not np.isnan(srcc) else 0.0
        plcc = float(plcc) if not np.isnan(plcc) else 0.0

    print(f"\nSRCC: {srcc:.4f}, PLCC: {plcc:.4f} ({len(predictions)}/{len(data)} valid)")
    print(f"MLLM assisted extractions: {mllm_assisted_count}")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['image_name']}: {r['error_message']}")
        print(f"{'='*80}\n")

    result_data = {
        'srcc': srcc,
        'plcc': plcc,
        'valid_count': len(predictions),
        'total_count': len(data),
        'mllm_assisted_count': mllm_assisted_count,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_authenticity_detection(inferencer, data_path, image_dir, output_path, num_demos=3):
    """Public release documentation."""
    print(f"\n=== Evaluating Forgery Detection ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    correct = 0
    total = 0
    mllm_assisted_count = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Forgery Detection {num_demos}shot"):
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        demos = item.get('demos', [])[:num_demos] if num_demos > 0 else []


        input_list = []
        for demo in demos:
            demo_img_path = os.path.join(image_dir, demo['image_name'])


            if not os.path.exists(demo_img_path):
                raise FileNotFoundError(f"Forgery Detection demo image required but not found: {demo_img_path}")

            demo_img = Image.open(demo_img_path).convert("RGB")
            demo_instruction = get_instruction_text(demo)
            demo_answer = demo.get('answer', demo.get('annotation', ''))


            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_instruction)
            input_list.append(f"\nAssistant: {demo_answer}")

        target_img = Image.open(image_path).convert("RGB")
        target_instruction = get_instruction_text(item)
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(target_img)
        input_list.append(target_instruction)
        input_list.append("\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for {item['image_name']}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        answer_str = extract_answer_from_tags(prediction)


        if answer_str is None:
            answer_str = mllm_assisted_extraction(
                prediction,
                target_instruction,
                "Either 'Real' or 'Fake'",
                options=["Real", "Fake"]
            )
            if answer_str:
                mllm_assisted_count += 1


        pred_label = None
        if answer_str:
            answer_lower = answer_str.strip().lower()

            if 'real' in answer_lower or 'authentic' in answer_lower or 'genuine' in answer_lower:
                pred_label = 'real'
            elif 'fake' in answer_lower or 'synthetic' in answer_lower or 'generated' in answer_lower:
                pred_label = 'fake'

            elif len(answer_str) == 1 and answer_str.upper() in "ABCD":
                if 'correct_label' in item:
                    pred_label = answer_str.upper()


        gt_label = extract_label_from_annotation(item.get('answer', item.get('annotation', '')))
        if 'correct_label' in item:
            gt_label = item['correct_label']

        is_correct = (pred_label == gt_label) if pred_label is not None else False
        if pred_label is not None:
            total += 1
            if is_correct:
                correct += 1

        results.append({
            'image_name': item['image_name'],
            'question': get_instruction_text(item),
            'ground_truth': gt_label,
            'prediction': prediction,
            'extracted_answer': answer_str,
            'pred_label': pred_label,
            'correct': is_correct,
            'mllm_assisted': answer_str is not None and extract_answer_from_tags(prediction) is None,
            'inference_failed': inference_failed,
            'error_message': error_message
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"MLLM assisted extractions: {mllm_assisted_count}")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['image_name']}: {r['error_message']}")
        print(f"{'='*80}\n")

    result_data = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'mllm_assisted_count': mllm_assisted_count,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_image_perfection(inferencer, data_path, image_dir, output_path, num_demos=3, qalign_model=None, gen_output_dir=None, skip_scoring=False, degraded_dir=None, gt_dir=None):
    """Public release documentation."""

    actual_degraded_dir = degraded_dir if degraded_dir is not None else image_dir
    actual_gt_dir = gt_dir if gt_dir is not None else image_dir

    if skip_scoring:
        print(f"\n=== Generating Images for Visual Refinement ({num_demos}-shot) ===")
        print("⚠️  Scoring skipped - use score_perfection_images.py to score later")
    else:
        print(f"\n=== Evaluating Visual Refinement with Q-Align ({num_demos}-shot) ===")

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]


    if gen_output_dir is None:
        gen_output_dir = os.path.join(os.path.dirname(output_path), "visual_refinement_generated")
    os.makedirs(gen_output_dir, exist_ok=True)

    results = []
    total_efficiency = 0.0
    valid_count = 0

    inference_params = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc=f"Visual Refinement {num_demos}shot"):

        input_img_rel = item.get('image_name', '')
        input_img_path = os.path.join(actual_degraded_dir, input_img_rel) if input_img_rel else ''

        gt_img_rel = item.get('answer', '')
        gt_img_path = os.path.join(actual_gt_dir, gt_img_rel) if gt_img_rel else ''

        if not os.path.exists(input_img_path):
            raise FileNotFoundError(f"Input image not found: {input_img_path}")

        demos = item.get('demos', [])[:num_demos] if num_demos > 0 else []



        input_list = []
        for demo in demos:

            demo_src_rel = demo.get('image_name', '')
            demo_src_path = os.path.join(actual_degraded_dir, demo_src_rel) if demo_src_rel else ''


            demo_gt_rel = demo.get('answer', '')
            demo_gt_path = os.path.join(actual_gt_dir, demo_gt_rel) if demo_gt_rel else ''


            if not demo_src_path:
                raise ValueError(f"Visual Refinement demo source path is empty for demo: {demo}")

            if not os.path.exists(demo_src_path):
                raise FileNotFoundError(f"Visual Refinement demo source image required but not found: {demo_src_path}")

            if not demo_gt_path:
                raise ValueError(f"Visual Refinement demo GT path is empty for demo: {demo}")

            if not os.path.exists(demo_gt_path):
                raise FileNotFoundError(f"Visual Refinement demo GT image required but not found: {demo_gt_path}")


            demo_img = Image.open(demo_src_path).convert("RGB")
            demo_instruction = demo.get('instruction', demo.get('text', ''))
            demo_gt_img = Image.open(demo_gt_path).convert("RGB")

            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_instruction)
            input_list.append("\nAssistant: ")
            input_list.append(demo_gt_img)


        source_img = Image.open(input_img_path).convert("RGB")
        target_instruction = item.get('instruction', item.get('text', ''))
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(source_img)
        input_list.append(target_instruction)
        input_list.append("\nAssistant:")


        sample_id = item.get('id', item.get('sample_id', os.path.basename(item.get('image_name', 'unknown')).replace('.png', '')))

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=False,
                think=False,
                **inference_params
            )
            generated_img = output_list[-1]


            gen_img_path = os.path.join(gen_output_dir, f"perfected_{sample_id}.png")
            if isinstance(generated_img, Image.Image):
                generated_img.save(gen_img_path)
            else:
                gen_img_path = None
        except Exception as e:
            print(f"Error perfecting {sample_id}: {e}")
            gen_img_path = None


        s_in = item.get('score_l')
        s_GT = item.get('score_h')


        s_out = None
        s_out_quality = None
        s_out_aesthetics = None

        if not skip_scoring:

            if gen_img_path and os.path.exists(gen_img_path) and qalign_model:
                from utils.scoring import compute_qalign_score
                score_dict = compute_qalign_score(gen_img_path, qalign_model)
                if score_dict:
                    s_out = score_dict['total_score']
                    s_out_quality = score_dict['quality_score']
                    s_out_aesthetics = score_dict['aesthetics_score']


        efficiency = None
        if not skip_scoring:
            if s_in is not None and s_out is not None:
                denominator = 5.0 - s_in
                if abs(denominator) > 1e-6:
                    efficiency = (s_out - s_in) / denominator * 100
                    total_efficiency += efficiency
                    valid_count += 1

        results.append({
            'sample_id': sample_id,
            'instruction': target_instruction,
            'input_image_path': input_img_path,
            'gt_image_path': gt_img_path,
            'generated_path': gen_img_path,
            'qalign_score_input': s_in,
            'qalign_score_output': s_out,
            'qalign_score_output_quality': s_out_quality,
            'qalign_score_output_aesthetics': s_out_aesthetics,
            'qalign_score_gt': s_GT,
            'efficiency': efficiency
        })

    mean_efficiency = total_efficiency / valid_count if valid_count > 0 else 0.0

    if skip_scoring:
        print(f"\n✅ Image generation completed!")
        print(f"Generated images saved to: {gen_output_dir}")
        print(f"Total generated: {len([r for r in results if r['generated_path']])} / {len(data)}")
        print(f"\n⏭️  Next step: Run scoring with:")
        print(f"   python score_perfection_images.py \\")
        print(f"       --generated-dir {gen_output_dir} \\")
        print(f"       --data-path {data_path} \\")
        print(f"       --output-path {output_path}")
    else:
        print(f"\nMean Efficiency: {mean_efficiency:.2f}% ({valid_count}/{len(data)} valid)")

    result_data = {
        'mean_efficiency': mean_efficiency if not skip_scoring else None,
        'valid_count': valid_count,
        'total_count': len(data),
        'scoring_skipped': skip_scoring,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    if skip_scoring:
        print(f"\nGeneration manifest saved to {output_path}")
    else:
        print(f"Results saved to {output_path}")
    return results


def _extract_label_from_text(text, options):
    """Public release documentation."""
    if not text:
        return None

    lower_text = text.lower()


    sorted_options = sorted(options, key=lambda x: len(x), reverse=True)

    for opt in sorted_options:
        if opt.lower() in lower_text:
            return opt

    return None


def eval_fcb_classification(inferencer, data_path, image_dir, output_path, num_demos=2):
    """Public release documentation."""
    print(f"\n=== Evaluating Fast Concept Mapping ===")

    with open(data_path, "r") as f:
        data = json.load(f)
    samples = data.get("samples", [])

    results = []
    correct = 0
    total = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for idx, item in enumerate(tqdm(samples, desc="Fast Concept Mapping")):
        query = item.get("query", {})
        demos = item.get("demos", [])[:num_demos] if num_demos > 0 else []
        options = item.get("meta", {}).get("options") or []
        gt_label = query.get("label", "")

        input_list = []

        for demo in demos:
            img_path = os.path.join(image_dir, demo.get("image", ""))


            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Fast Concept Mapping demo image required but not found: {img_path}")

            demo_img = Image.open(img_path).convert("RGB")
            demo_instruction = demo.get('instruction','')
            demo_label = demo.get('label','')


            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_instruction)
            input_list.append(f"\nAssistant: {demo_label}")

        query_img = os.path.join(image_dir, query.get("image", ""))
        if not os.path.exists(query_img):
            raise FileNotFoundError(f"Query image not found: {query_img}")

        query_image = Image.open(query_img).convert("RGB")
        query_instruction = query.get('instruction','')
        if len(demos) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")
        input_list.append(query_image)
        input_list.append(query_instruction)
        input_list.append("\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for Fast Concept Mapping sample: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)

        pred_label = _extract_label_from_text(prediction, options or [gt_label])
        is_correct = pred_label is not None and pred_label.lower() == gt_label.lower()
        total += 1
        if is_correct:
            correct += 1

        results.append({
            "id": item.get("meta", {}).get("item_id") or f"fcb_{idx}",
            "prediction": prediction,
            "pred_label": pred_label,
            "answer": gt_label,
            "correct": is_correct,
            "inference_failed": inference_failed,
            "error_message": error_message
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFast Concept Mapping Accuracy: {correct}/{total} = {accuracy:.4f}")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - {r['id']}: {r['error_message']}")
        print(f"{'='*80}\n")

    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_fci_t2i(inferencer, data_path, image_dir, output_path, num_demos=3, gen_output_dir=None):
    """Public release documentation."""
    print(f"\n=== Evaluating Fast Concept Generation ===")

    with open(data_path, "r") as f:
        data = json.load(f)
    samples = data.get("samples", [])

    if gen_output_dir is None:
        gen_output_dir = os.path.join(os.path.dirname(output_path), "fci_generated")
    os.makedirs(gen_output_dir, exist_ok=True)

    results = []
    total_score = 0.0
    valid_count = 0

    inference_params = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        do_sample=False,
        text_temperature=0.1,
    )

    for idx, item in enumerate(tqdm(samples, desc="Fast Concept Generation")):
        query = item.get("query", {})
        demos = item.get("demos", [])[:num_demos] if num_demos > 0 else []
        meta = item.get("meta", {})


        novel_label = query.get("label", "")
        structural_signature = query.get("structural_signature", "")
        instruction = query.get("instruction", "")

        base_label = ""
        for demo in demos:
            if not demo.get("is_novel", True):
                base_label = demo.get("label", "")
                break

        item_id = meta.get("item_id") or f"fci_{idx}"

        input_list = []
        for demo in demos:
            img_path = os.path.join(image_dir, demo.get("image", ""))


            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Fast Concept Generation demo image required but not found: {img_path}")

            demo_instruction = demo.get('instruction','')
            demo_img = Image.open(img_path).convert("RGB")


            input_list.append(f"\nUser: Generate an image: {demo_instruction}\nAssistant: ")
            input_list.append(demo_img)


        ref_img_path = os.path.join(image_dir, query.get("image", ""))
        if not os.path.exists(ref_img_path):
            raise FileNotFoundError(f"Reference image not found: {ref_img_path}")

        query_instruction = query.get('instruction','')
        if len(demos) > 0:
            input_list.append(f"\nUser: Generate an image: {query_instruction}\nAssistant:")
        else:
            input_list.append(f"User: Generate an image: {query_instruction}\nAssistant:")

        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=False,
                think=False,
                **inference_params
            )
            generated_img = output_list[-1]
            gen_path = os.path.join(gen_output_dir, f"{item_id}.jpg")
            if isinstance(generated_img, Image.Image):
                generated_img.save(gen_path)
            else:
                gen_path = None
        except Exception as e:
            print(f"Error generating Fast Concept Generation sample: {e}")
            gen_path = None


        category_consistency_score = 0.0
        aesthetic_score = 0.0
        text_alignment_score = 0.0
        final_score = 0.0
        rationale = ""

        if gen_path and os.path.exists(gen_path):
            judge_prompt = (
                "You are an expert image quality evaluator. You will see TWO images in order:\n\n"
                "**FIRST IMAGE = Generated Image**: This is the model's output that you need to evaluate.\n"
                "**SECOND IMAGE = Reference Image**: This shows what the target concept should look like.\n\n"
                "**IMPORTANT**: Do NOT confuse the two images. The FIRST image is what the model generated, the SECOND is the reference.\n\n"
                f"**Task Context**:\n"
                f"- Base Category: {base_label}\n"
                f"- Target Novel Category: {novel_label}\n"
                f"- Key Structural Feature: {structural_signature}\n"
                f"- Generation Instruction: {instruction}\n\n"
                "**Step 1**: First describe what you see in each image:\n"
                "- Generated (1st): [brief description]\n"
                "- Reference (2nd): [brief description]\n\n"
                "**Step 2**: Evaluate the GENERATED image on THREE dimensions (1-100 each):\n\n"
                "**1. Category Consistency (1-100)** - MOST IMPORTANT\n"
                f"Does the GENERATED image correctly represent '{novel_label}' with: {structural_signature}?\n"
                "- 85-100: Novel category clearly represented with key structural features\n"
                "- 70-84: Good representation, most features present with minor issues\n"
                "- 50-69: Recognizable attempt, but features incomplete or ambiguous\n"
                "- 25-49: Weak, mostly resembles base category, minimal novel features\n"
                "- 1-24: Wrong category or completely failed generation\n\n"
                "**2. Aesthetic Quality (1-100)**\n"
                "- 85-100: Professional quality, crisp details, no artifacts\n"
                "- 70-84: Good quality, minor flaws, clear overall\n"
                "- 50-69: Acceptable quality, some noticeable flaws\n"
                "- 25-49: Poor quality, significant artifacts or blur\n"
                "- 1-24: Very poor, major distortion or unintelligible\n\n"
                "**3. Text-Image Alignment (1-100)**\n"
                "- 85-100: All instruction elements accurately represented\n"
                "- 70-84: Most elements present with minor deviations\n"
                "- 50-69: Core elements present, some details missing\n"
                "- 25-49: Several elements missing or incorrect\n"
                "- 1-24: Minimal or no instruction following\n\n"
                "**Output Format**:\n"
                "Rationale: [Describe both images first, then explain your scores]\n"
                "Category_Consistency: [1-100]\n"
                "Aesthetic_Quality: [1-100]\n"
                "Text_Alignment: [1-100]"
            )


            judge_response = call_vllm_judge(judge_prompt, [gen_path, ref_img_path])

            try:

                rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Category_Consistency|$)', judge_response, re.IGNORECASE | re.DOTALL)
                if rationale_match:
                    rationale = rationale_match.group(1).strip()


                category_match = re.search(r'Category[_\s]*Consistency[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                aesthetic_match = re.search(r'Aesthetic[_\s]*Quality[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                alignment_match = re.search(r'Text[_\s]*Alignment[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)

                if category_match:
                    category_consistency_score = float(category_match.group(1))
                    category_consistency_score = min(100.0, max(1.0, category_consistency_score))

                if aesthetic_match:
                    aesthetic_score = float(aesthetic_match.group(1))
                    aesthetic_score = min(100.0, max(1.0, aesthetic_score))

                if alignment_match:
                    text_alignment_score = float(alignment_match.group(1))
                    text_alignment_score = min(100.0, max(1.0, text_alignment_score))


                if category_match and aesthetic_match and alignment_match:
                    final_score = (
                        category_consistency_score * 0.7 +
                        aesthetic_score * 0.15 +
                        text_alignment_score * 0.15
                    )
                    total_score += final_score
                    valid_count += 1
            except Exception as e:
                print(f"Error parsing scores: {e}")

        results.append({
            "id": item_id,
            "base_label": base_label,
            "novel_label": novel_label,
            "structural_signature": structural_signature,
            "instruction": instruction,
            "generated_path": gen_path,
            "reference_path": ref_img_path,
            "rationale": rationale,
            "category_consistency_score": category_consistency_score,
            "aesthetic_score": aesthetic_score,
            "text_alignment_score": text_alignment_score,
            "final_score": final_score
        })

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    print(f"\nMean Fast Concept Generation Score: {mean_score:.2f}/100 ({valid_count} valid)")

    with open(output_path, "w") as f:
        json.dump({
            "mean_score": mean_score,
            "valid_count": valid_count,
            "total_count": len(samples),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results



def eval_planning(inferencer, data_path, image_dir, output_path, num_demos=0):
    """Public release documentation."""
    print(f"\n=== Evaluating World-Aware Planning (MCQ) ===")

    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []
    correct = 0
    total = 0
    skipped = 0

    inference_params = dict(
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc="World-Aware Planning"):
        sample_id = item.get("id")
        image_paths = [os.path.join(image_dir, p) for p in item.get("images", [])]
        convs = item.get("conversations", [])



        task_instruction = ""
        history_steps = []  # [(image_idx, gpt_text), ...]
        mcq_question = ""
        gt_letter = ""
        img_idx = 0

        i = 0
        while i < len(convs):
            conv = convs[i]

            if conv['from'] in ['human', 'User'] and i == 0:

                task_instruction = conv['value']
            elif conv['from'] in ['human', 'User'] and 'Options:' in conv['value']:

                mcq_question = conv['value']
            elif conv['from'] == 'observation' and conv['value'] == '<image>':

                if i + 1 < len(convs):
                    next_conv = convs[i + 1]
                    if next_conv['from'] in ['gpt', 'Assistant']:

                        history_steps.append({
                            'image_idx': img_idx,
                            'gpt_text': next_conv['value']
                        })
                    elif next_conv['from'] in ['human', 'User'] and 'Options:' in next_conv['value']:


                        history_steps.append({
                            'image_idx': img_idx,
                            'gpt_text': None
                        })
                img_idx += 1
            elif conv['from'] in ['gpt', 'Assistant'] and i == len(convs) - 1:

                gpt_answer = conv['value']

                action_match = re.search(r'<action>\s*([A-D])\s*</action>', gpt_answer, re.IGNORECASE)
                if action_match:
                    gt_letter = action_match.group(1).upper()
            i += 1


        if not task_instruction or not mcq_question or not gt_letter:
            skipped += 1
            continue


        input_list = []
        missing = False


        input_list.append(f"User: {task_instruction}\nAssistant: Sure!")


        for step in history_steps[:-1]:
            img_path = image_paths[step['image_idx']] if step['image_idx'] < len(image_paths) else None
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")

                input_list.append("\nUser: ")
                input_list.append(img)
                if step['gpt_text']:
                    input_list.append(f"\nAssistant: {step['gpt_text']}")
            else:
                missing = True
                break

        if missing:
            raise FileNotFoundError(f"Missing history image for planning id {sample_id}")


        last_step = history_steps[-1] if history_steps else None
        if not last_step:
            raise ValueError(f"Missing last step for planning id {sample_id}")


        last_img_idx = min(last_step['image_idx'], len(image_paths) - 1) if image_paths else -1
        last_img_path = image_paths[last_img_idx] if last_img_idx >= 0 else None
        if not last_img_path or not os.path.exists(last_img_path):
            raise FileNotFoundError(f"Missing last image for planning id {sample_id}: {last_img_path}")

        last_img = Image.open(last_img_path).convert("RGB")


        input_list.append("\nUser: ")
        input_list.append(last_img)
        input_list.append(mcq_question)
        input_list.append("\nAssistant:")


        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False,
                **inference_params
            )
            prediction = output_list[-1].strip()
            inference_failed = False
            error_message = None
        except Exception as e:
            print(f"[WARNING] Inference failed for planning id {sample_id}: {e}")
            prediction = ""
            inference_failed = True
            error_message = str(e)


        pred_letter = ""

        pred_action_match = re.search(r'<action>\s*([A-D])\s*</action>', prediction, re.IGNORECASE)
        if pred_action_match:
            pred_letter = pred_action_match.group(1).upper()
        else:

            print(f"  [MLLM Assist] No strict format match. Calling Judge...")
            print(f"  Prediction: {prediction[:100]}...")
            judge_prompt = (
                "You are an expert evaluator for multiple choice questions.\n"
                "Below is a question and a model's predicted answer.\n\n"
                f"**Question**:\n{mcq_question}\n\n"
                f"**Model Prediction**:\n{prediction}\n\n"
                "**Task**:\n"
                "Determine which option (A, B, C, or D) the Model Prediction corresponds to.\n"
                "If the model prediction is synonymous with one of the options, select that option.\n"
                "If the model prediction is unrelated or wrong, select 'None'.\n\n"
                "Output ONLY the single letter (A, B, C, D) or None."
            )

            judge_response = call_vllm_judge(judge_prompt)
            judge_match = re.search(r'\b([A-D])\b', judge_response, re.IGNORECASE)
            if judge_match:
                pred_letter = judge_match.group(1).upper()
                print(f"  [MLLM Assist] Judge assigned: {pred_letter}")
            else:
                print(f"  [MLLM Assist] Judge failed. Response: {judge_response}")


        is_correct = pred_letter == gt_letter

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "id": sample_id,
            "task_instruction": task_instruction,
            "mcq_question": mcq_question,
            "gt_letter": gt_letter,
            "pred_letter": pred_letter,
            "prediction_full": prediction,
            "correct": is_correct,
            "inference_failed": inference_failed,
            "error_message": error_message
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\nWorld-Aware Planning Accuracy (MCQ): {correct}/{total} = {accuracy:.4f}")
    if skipped > 0:
        print(f"Skipped {skipped} samples")


    failed_count = sum(1 for r in results if r.get('inference_failed', False))
    if failed_count > 0:
        failure_rate = failed_count / len(results)
        print(f"\n{'='*80}")
        print(f"Inference Failure Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Failure rate: {failure_rate:.2%}")
        if failure_rate > 0.1:
            print(f"\n⚠️  WARNING: High failure rate ({failure_rate:.2%})!")
            print("Failed samples:")
            for r in results:
                if r.get('inference_failed'):
                    print(f"  - id={r['id']}: {r['error_message']}")
        print(f"{'='*80}\n")

    with open(output_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "skipped": skipped,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_visualcloze_g(inferencer, data_path, image_dir, output_path, num_demos=8, gen_output_dir=None, **kwargs):
    """Public release documentation."""
    # Ignore legacy kwargs (e.g., eval_mode) to keep compatibility with existing callers.
    _ = kwargs

    print(f"\n=== Evaluating Analogical Editing ({num_demos}-shot, mode=mllm_judge) ===")

    if str(data_path).endswith(".jsonl"):
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)

    if gen_output_dir is None:
        gen_output_dir = os.path.join(os.path.dirname(output_path), "visualcloze_generated")
    os.makedirs(gen_output_dir, exist_ok=True)

    results = []
    total_score = 0.0
    valid_count = 0

    inference_params = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        do_sample=False,
        text_temperature=0.1,
    )

    def _parse_task_name(task_name: str):
        parts = task_name.split('_') if isinstance(task_name, str) and task_name else []
        if len(parts) < 2:
            return [], None
        return parts[:-1], parts[-1]

    def _get_instruction(sample: dict) -> str:
        if not isinstance(sample, dict):
            return ""
        inst = sample.get("instruction", "")
        return inst.strip() if isinstance(inst, str) else ""

    def _get_intent_text(item: dict) -> str:
        if not isinstance(item, dict):
            return ""
        intent = item.get("intent", {})
        if not isinstance(intent, dict):
            return ""
        structured_intent = {
            "edit_type": intent.get("edit_type", ""),
            "object": intent.get("object", ""),
            "from": intent.get("from", ""),
            "to": intent.get("to", ""),
        }
        if not any(str(v).strip() for v in structured_intent.values()):
            return ""
        return json.dumps(structured_intent, ensure_ascii=False)

    def _resolve_img_path(path_str: str, sample_id) -> str:
        """Resolve image path for both raw dataset layout and copied benchmark layout."""
        if not isinstance(path_str, str) or not path_str:
            return ""
        raw = Path(path_str)
        if raw.is_absolute():
            return str(raw)

        base = Path(image_dir)
        candidates = [
            base / raw,                              # e.g. /root/images/T2I/xxx.png
            base / str(sample_id) / raw,            # e.g. /root/<id>/images/T2I/xxx.png
            base / "Gen" / raw,                     # e.g. /root/Gen/images/T2I/xxx.png
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        # Return the first candidate for downstream error messages.
        return str(candidates[0])

    skipped_count = 0

    for item in tqdm(data, desc=f"Analogical Editing {num_demos}shot"):
        item_id = item.get('id')
        task_name = str(item.get('task_name', ''))
        all_demos = item.get('demo', [])
        if not isinstance(all_demos, list):
            print(f"[Analogical Editing] Skip id={item_id}: 'demo' must be a list")
            skipped_count += 1
            continue
        demos = all_demos[:num_demos] if num_demos > 0 else []
        query = item.get('query', {})
        if not isinstance(query, dict):
            print(f"[Analogical Editing] Skip id={item_id}: 'query' must be a dict")
            skipped_count += 1
            continue


        input_fields, output_field = _parse_task_name(task_name)
        if not output_field:
            print(f"[Analogical Editing] Skip id={item_id}: invalid task_name '{task_name}'")
            skipped_count += 1
            continue
        if not input_fields:
            print(f"[Analogical Editing] Skip id={item_id}: no input fields parsed from task_name '{task_name}'")
            skipped_count += 1
            continue



        input_list = []

        for demo in demos:
            if not isinstance(demo, dict):
                continue

            demo_pack = ["\nUser: "]
            demo_valid = True


            for key in input_fields:
                if key not in demo or not isinstance(demo[key], str) or not demo[key]:
                    demo_valid = False
                    break
                demo_input_path = _resolve_img_path(demo[key], item_id)
                if not os.path.exists(demo_input_path):
                    demo_valid = False
                    break
                demo_pack.append(Image.open(demo_input_path).convert("RGB"))


            if output_field not in demo or not isinstance(demo[output_field], str) or not demo[output_field]:
                demo_valid = False

            if demo_valid:
                demo_output_path = _resolve_img_path(demo[output_field], item_id)
                if not os.path.exists(demo_output_path):
                    demo_valid = False

            if not demo_valid:
                continue

            demo_text = _get_instruction(demo)
            if demo_text:
                demo_pack.append(demo_text)


            demo_pack.append("\nAssistant: ")
            demo_pack.append(Image.open(demo_output_path).convert("RGB"))
            input_list.extend(demo_pack)


        if len(input_list) > 0:
            input_list.append("\nUser: ")
        else:
            input_list.append("User: ")


        query_input_paths = []
        missing = False
        for key in input_fields:
            v = query.get(key, "")
            if not isinstance(v, str) or not v:
                missing = True
                break
            query_input_path = _resolve_img_path(v, item_id)
            if not os.path.exists(query_input_path):
                missing = True
                break
            query_input_paths.append(query_input_path)
            input_list.append(Image.open(query_input_path).convert("RGB"))

        query_text = _get_instruction(query)
        if query_text:
            input_list.append(query_text)
        judge_text = _get_intent_text(item)


        if output_field not in query or not isinstance(query.get(output_field), str) or not query.get(output_field):
            missing = True
        else:
            query_output_path = _resolve_img_path(query[output_field], item_id)
            if not os.path.exists(query_output_path):
                missing = True

        if missing:
            print(f"[Analogical Editing] Skip id={item_id}: missing query images/output")
            skipped_count += 1
            continue
        if not judge_text:
            print(f"[Analogical Editing] Skip id={item_id}: missing intent for judge")
            skipped_count += 1
            continue


        input_list.append("\nAssistant:")


        try:
            output_list = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=False,
                think=False,
                **inference_params
            )
            generated_img = output_list[-1]

            gen_img_path = os.path.join(gen_output_dir, f"visualcloze_{item_id}.jpg")
            if isinstance(generated_img, Image.Image):
                generated_img.save(gen_img_path)
            else:
                gen_img_path = None
        except Exception as e:
            print(f"Error processing Analogical Editing sample {item_id}: {e}")
            gen_img_path = None


        edit_accuracy_score = 0.0
        preservation_score = 0.0
        final_score = 0.0
        judge_response = ""
        rationale = ""
        source_img_path = query_input_paths[0] if len(query_input_paths) > 0 else None

        if source_img_path and gen_img_path and os.path.exists(gen_img_path) and os.path.exists(query_output_path):
            judge_prompt = (
                "You are an expert image editing evaluator.\n\n"
                "**You will see 3 images in order:**\n"
                "  1. Original Image - The source image before editing\n"
                "  2. Reference Image (GT) - The ground-truth edited result\n"
                "  3. Model Output - The model's edited output\n\n"
                f"**Edit Intent (structured)**: {judge_text}\n\n"
                "Important: The edit instruction can be broad (e.g., style transfer).\n"
                "Use the Reference Image to identify the specific transformation target.\n"
                "Judge whether Model Output captures that specific transformation, not just a generic edit.\n\n"
                "Evaluate on TWO dimensions. Score each from 1-100. DO NOT calculate the total score.\n\n"
                "**1. Edit Accuracy (1-100)** - How accurately does output realize the intended edit?\n"
                "- 90-100: Specific transformation matches GT very closely; key edited attributes are correct\n"
                "- 75-89: Core transformation correct; minor mismatches in details/intensity/location\n"
                "- 55-74: Partially correct; noticeable mismatch with GT transformation target\n"
                "- 25-54: Weak execution; major misunderstanding of the intended transformation\n"
                "- 1-24: Wrong edit or largely no meaningful edit\n\n"
                "**2. Preservation & Quality (1-100)** - Are non-edited regions preserved with good quality?\n"
                "- 90-100: Non-target regions well preserved, high fidelity, minimal artifacts\n"
                "- 75-89: Minor unintended changes, quality mostly good\n"
                "- 55-74: Noticeable unintended changes or quality issues\n"
                "- 25-54: Significant preservation failure and/or clear degradation\n"
                "- 1-24: Severe corruption or widespread unintended alterations\n\n"
                "**Output Format**:\n"
                "Rationale: [Brief assessment]\n"
                "Edit_Accuracy: [1-100]\n"
                "Preservation_Quality: [1-100]"
            )

            # For this task, evaluate with original + GT + generated only.
            images_to_judge = [source_img_path, query_output_path, gen_img_path]
            judge_response = call_vllm_judge(judge_prompt, images_to_judge)

            try:
                rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Edit_Accuracy|$)', judge_response, re.IGNORECASE | re.DOTALL)
                if rationale_match:
                    rationale = rationale_match.group(1).strip()

                accuracy_match = re.search(r'Edit[_\s]*Accuracy[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                preservation_match = re.search(r'Preservation[_\s]*(?:Quality|&\s*Quality)?[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                if accuracy_match:
                    edit_accuracy_score = min(100.0, max(1.0, float(accuracy_match.group(1))))
                if preservation_match:
                    preservation_score = min(100.0, max(1.0, float(preservation_match.group(1))))

                # Weighted score: Edit Accuracy 70%, Preservation 30%.
                if accuracy_match and preservation_match:
                    final_score = edit_accuracy_score * 0.7 + preservation_score * 0.3
                    total_score += final_score
                    valid_count += 1
            except Exception as e:
                print(f"Error parsing Analogical Editing judge scores for id {item_id}: {e}")

        results.append({
            'id': item_id,
            'task_name': task_name,
            'num_demos': len(demos),
            'instruction': judge_text,
            'query_instruction': query_text,
            'source_path': source_img_path,
            'query_input': query_input_paths,
            'query_reference': query_output_path,
            'generated_path': gen_img_path,
            'eval_mode': 'i2i_style_mllm_judge',
            'dinov3_similarity': None,
            'edit_accuracy_score': edit_accuracy_score,
            'preservation_quality_score': preservation_score,
            'rationale': rationale,
            'final_score': final_score
        })

    mean_score = total_score / valid_count if valid_count > 0 else 0.0
    print(f"\nMean Analogical Editing Score: {mean_score:.2f}/100 ({valid_count}/{len(data)} valid, skipped={skipped_count})")

    result_data = {
        'mean_score': mean_score,
        'valid_count': valid_count,
        'total_count': len(data),
        'skipped_count': skipped_count,
        'results': results
    }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_visualcloze_u(inferencer, data_path, image_dir, output_path, num_demos=8, gen_output_dir=None):
    """Public release documentation."""
    print(f"\n=== Evaluating Analogical Inference ({num_demos}-shot) ===")


    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    results = []
    total_intent_score = 0.0
    total_correctness_score = 0.0
    valid_count = 0

    # Intent definitions for MLLM judge - 10-task configuration (4 keyword + 6 instruction-free)
    INTENT_DEFINITIONS = {
        # Keyword instruction tasks
        "bbox_specific": {
            "definition": "Detect bounding box for a specific object type. Input: object name.",
            "format_requirements": "Output MUST be a valid JSON string `{\"object_name\": [x1, y1, x2, y2]}`. Coordinates must be normalized floats (0.0-1.0). strictly [x1, y1, x2, y2].",
            "content_requirements": "The bounding box must tightly enclose the specified object. It must not cover unrelated objects.",
            "intent_anchors": """100: Perfect `{\"key\": [x1,y1,x2,y2]}` JSON format.
80: Valid JSON but minor key mismatch (e.g., using 'box' instead of object name).
60: Correct coordinates found but in plain text format (e.g., '0.1, 0.2...').
40: Textual description of location (e.g., 'top left') without coordinates.
20: Complete failure to output bounding box data (prose only).""",
            "correctness_anchors": """100: High overlap (IoU > 0.8) with Ground Truth structure.
80: Good overlap (IoU > 0.5) covering the object properly.
60: Partial overlap (IoU > 0.3) or loose box containing background.
40: Wrong object detected or very poor localization (IoU < 0.1).
20: Coordinates encompass entire image (0,0,1,1) or are clearly hallucinations."""
        },
        
        "object_material_groups": {
            "definition": "List objects made of a specific material.",
            "format_requirements": "Comma-separated object names (e.g., `fork, spoon`). No brackets, no quotes, no numbering.",
            "content_requirements": "Must list ALL objects visible made of that material. No hallucinations.",
            "intent_anchors": """100: Clean comma-separated list (e.g., `obj1, obj2`).
80: List with minor formatting noise (e.g., bullets or `1. obj`).
60: Conversational list (e.g., `The objects are obj1 and obj2`).
40: Long paragraph description mentioning materials.
20: Irrelevant format (e.g., coordinate output).""",
            "correctness_anchors": """100: Perfectly identified all objects of the target material.
80: Identified main objects but missed 1 minor/ambiguous item.
60: Included correct objects but also 1 incorrect object (hallucination).
40: Missed majority of objects or included multiple wrong items.
20: Listed specific objects that are NOT made of that material."""
        },
        
        "object_state_groups": {
            "definition": "List objects with a specific state (e.g., open, closed, broken).",
            "format_requirements": "Comma-separated object names. No extra text.",
            "content_requirements": "Accurate identification of object states.",
            "intent_anchors": """100: Clean comma-separated list.
80: List with extra punctuation or minor syntax issues.
60: Sentence-based listing (e.g., `I see a door and a window`).
40: Detailed description of object states instead of a list.
20: Unrelated response type.""",
            "correctness_anchors": """100: Accurately lists only objects in the requested state.
80: Correct concepts but loose object naming (e.g., 'cupboard' vs 'cabinet').
60: Mixed bag: some correct, some wrong state.
40: Mostly wrong objects (e.g., listing 'closed' items for 'open').
20: Complete failure to identify state."""
        },
        
        "object_spatial_locations": {
            "definition": "Describe spatial location of a specific object.",
            "format_requirements": "Natural language description (short phrase). No JSON, no coordinates.",
            "content_requirements": "Accurate relative position (left/right, on top of, etc.).",
            "intent_anchors": """100: Concise phrase (e.g., `top right corner`).
80: Complete sentence (e.g., `The object is in the top right`).
60: Overly verbose description of the scene context.
40: Outputting coordinates or bounding boxes instead of text.
20: Outputting object attributes instead of location.""",
            "correctness_anchors": """100: Precise and accurate location description.
80: Generally correct but slightly vague (e.g., `right` instead of `top right`).
60: Technically true but misleading (e.g., `on the table` when many items are).
40: Wrong relative position (e.g., `left` instead of `right`).
20: Completely hallucinated location or object."""
        },
        
        # Instruction-free tasks
        "object_attributes": {
            "definition": "Describe attributes of specific objects including color, material, size, and state.",
            "format_requirements": "Format: `object_name(attr1=val1, attr2=val2); ...`. Semicolon separator.",
            "content_requirements": "Correct attributes for each object. Do not hallucinate objects.",
            "intent_anchors": """100: Strict `obj(k=v)` syntax.
80: Minor syntax deviations (e.g., using commas instead of semicolons).
60: Structured output like JSON or key-value pairs per line.
40: Unstructured descriptive sentences.
20: Just a list of objects without attributes.""",
            "correctness_anchors": """100: All attribute values (color, material, etc.) are visually correct.
80: Most attributes correct, 1 minor error (e.g., `dark blue` vs `black`).
60: Some correct attributes, but major error on key features (e.g., wrong material).
40: Attributes largely generic or incorrect.
20: Hallucinated object properties entirely."""
        },
        
        "object_inventory": {
            "definition": "List all objects present in the image with their counts.",
            "format_requirements": "Format: `object_name×count`, comma-separated. Use `×` symbol.",
            "content_requirements": "Correct counts for visible objects. Missing small objects is penalized less than hallucinating.",
            "intent_anchors": """100: Strict `obj×N` syntax with correct symbol.
80: Using `x` or `*` instead of `×` (e.g., `dogx2`).
60: Textual counts (e.g., `2 dogs, 1 cat`) or `obj: N`.
40: List of objects without counts.
20: Standard captioning or description.""",
            "correctness_anchors": """100: Exact counts for all main objects.
80: Counts off by one for numerous items (>5).
60: Missed several small objects or counted 1-2 extra.
40: Major counting failure (e.g., saying 5 when there are 2).
20: Hallucinating objects that do not exist."""
        },
        
        "scene_attributes": {
            "definition": "Identify scene attributes (category, weather, time, lighting, season, crowdedness).",
            "format_requirements": "Format: semicolon-separated `key=value` pairs.",
            "content_requirements": "Attributes must accurately reflect the scene.",
            "intent_anchors": """100: Strict `key=value;` syntax.
80: Using commas or newlines instead of semicolons.
60: JSON format or specific list format.
40: Prose/Paragraph description of the scene.
20: Outputting irrelevant keys or data types.""",
            "correctness_anchors": """100: Accurately captures environmental conditions.
80: 1 attribute slightly off (e.g., `afternoon` vs `morning`).
60: 2+ attributes incorrect (e.g., wrong season).
40: Major contradiction (e.g., `sunny` for `night`).
20: Describing the wrong scene entirely."""
        },
        
        "scene_mood": {
            "definition": "Extract scene atmosphere and mood.",
            "format_requirements": "Format: `lighting: val, weather: val, ...` (comma-separated with colons).",
            "content_requirements": "Use correct descriptors from list: [natural, artificial, sunny, cloudy, etc.]",
            "intent_anchors": """100: Strict `Key: Value, Key: Value` format.
80: Minor delimiters change (e.g., semicolons) or capitalization.
60: Only values listed without keys.
40: Sentences describing mood.
20: Irrelevant keywords.""",
            "correctness_anchors": """100: Mood descriptors perfectly match visual atmosphere.
80: Plausible but alternative interpretation for subjective fields.
60: Lighting or Weather is factually wrong.
40: Mood implies opposite emotion/tone of image.
20: Random words unrelated to the image."""
        },
        
        "relations": {
            "definition": "Identify spatial or semantic relationships between objects.",
            "format_requirements": "Format: `(subject, predicate, object); ...` Tuple style.",
            "content_requirements": "True relations visible in image.",
            "intent_anchors": """100: Strict `(s, p, o)` tuple format.
80: Missing parentheses but structured `s p o`.
60: Natural language phrases (e.g., `s is p o`).
40: Just a list of objects or a caption.
20: JSON or Code format.""",
            "correctness_anchors": """100: Relationships are physically accurate (A is truly on B).
80: Relation is valid but generic (e.g., `near` instead of `left of`).
60: Reversed subject/object (e.g., `Horse riding Man`).
40: Hallucinated relationship between unconnected objects.
20: Hallucinated objects in relations."""
        },
        
        "caption_styled": {
            "definition": "Generate a stylized caption matching the artistic style of demonstrations.",
            "format_requirements": "Natural language sentences matching specific style (minimalist, vivid, etc.).",
            "content_requirements": "Accurate description AND correct style tone.",
            "intent_anchors": """100: Sentence structure and length match the requested style.
80: Correct length but style is generic.
60: Extremely short or long compared to demos.
40: List of keywords instead of caption.
20: Technical metadata instead of caption.""",
            "correctness_anchors": """100: Perfect blend of image accuracy and stylistic tone.
80: Accurate content but style is weak.
60: Stylized well but misses main subject content.
40: Factually incorrect description of the image.
20: Generic text applicable to any image."""
        }
    }

    for item in tqdm(data, desc=f"Analogical Inference {num_demos}shot"):
        image_name = item.get('image_name')
        intent = item.get('intent', '')
        text = item.get('text', '')  # Keyword instruction (e.g., 'poster', 'metal', 'open')
        answer = item.get('answer', item.get('annotation', ''))
        demos = item.get('demos', [])[:num_demos] if num_demos > 0 else []

        # Parse base intent and suffix
        intent_parts = intent.split(':', 1)
        base_intent = intent_parts[0]
        intent_suffix = intent_parts[1] if len(intent_parts) > 1 else None

        # Check if image exists
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image missing: {image_name}")

        # Build input list for ICL with proper User/Assistant markers
        # Keyword mode: User: <image> keyword \nAssistant: Answer
        # Instruction-free mode: User: <image> \nAssistant: Answer
        input_list = []
        missing = False

        # Add demo examples with clearer formatting
        for idx, demo in enumerate(demos):
            demo_image_name = demo.get('image_name')
            demo_text = demo.get('text', '')  # Demo keyword instruction
            demo_answer = demo.get('answer', demo.get('annotation', ''))
            demo_path = os.path.join(image_dir, demo_image_name)

            if not os.path.exists(demo_path):
                missing = True
                break



            if idx > 0:
                input_list.append("\n")
            
            input_list.append("User: ")
            input_list.append(Image.open(demo_path).convert("RGB"))
            

            if demo_text:
                input_list.append(f" {demo_text}")
            
            input_list.append(f"\nAssistant: {demo_answer}")

        if missing:
            raise FileNotFoundError(f"Demo image missing for {image_name}")

        # Add query image with User/Assistant format

        if len(demos) > 0:
            input_list.append("\n")
        
        input_list.append("User: ")
        input_list.append(Image.open(image_path).convert("RGB"))
        

        if text:
            input_list.append(f" {text}")
        
        input_list.append("\nAssistant:")

        # Generate answer using model
        try:
            output_text = inferencer.interleave_inference(
                input_lists=input_list,
                understanding_output=True,
                think=False
            )
            if isinstance(output_text, list) and len(output_text) > 0:
                generated_answer = output_text[0] if isinstance(output_text[0], str) else str(output_text[0])
            else:
                generated_answer = str(output_text)

            # Clean up the generated answer (remove any User:/Assistant: markers if present)
            generated_answer = generated_answer.replace("User:", "").replace("Assistant:", "").strip()

        except Exception as e:
            print(f"Error generating answer for {image_name}: {e}")
            generated_answer = ""

        # Use MLLM to evaluate the generated answer
        intent_score = 0.0
        correctness_score = 0.0
        rationale = ""

        if generated_answer:
            # Get intent task info
            task_info = INTENT_DEFINITIONS.get(base_intent, {})
            # Fallback for unknown intent
            if not task_info:
                 task_info = {
                     "definition": "Visual understanding task.",
                     "format_requirements": "Follow the format shown in the ground truth.",
                     "content_requirements": "Answer must be factually correct.",
                     "intent_anchors": "High for correct format, Low for wrong format.",
                     "correctness_anchors": "High for accurate content, Low for significant errors."
                 }

            # Build judge prompt with specific anchors
            # Explicitly instruct constraints per intent type
            judge_prompt = (
                "You are evaluating a model's performance on a visual in-context learning task.\n"
                "The model learned from demonstrations (8 examples) and generated an answer for a new query.\n\n"
                
                "=== TASK INFORMATION ===\n"
                f"Task Name: {intent}\n"
                f"Definition: {task_info.get('definition', '')}\n"
                f"Required Format: {task_info.get('format_requirements', '')}\n"
                f"Content Requirements: {task_info.get('content_requirements', '')}\n\n"
                
                f"=== ANSWERS TO EVALUATE ===\n"
                f"Ground Truth: {answer}\n"
                f"Model Output: {generated_answer}\n\n"
                
                "=== EVALUATION CRITERIA ===\n"
                "You must evaluate on TWO SEPARATE dimensions:\n\n"
                
                "1️⃣ INTENT SCORE (0-100): Task Understanding & Format Compliance\n"
                f"   • Did the model follow: {task_info.get('format_requirements', '')}?\n"
                "   • Format violations are CRITICAL.\n"
                "   Scoring Anchors:\n"
                f"   {task_info.get('intent_anchors', '')}\n\n"
                
                "2️⃣ CORRECTNESS SCORE (1-100): Content Accuracy\n"
                "   • How accurate is the actual content compared to ground truth?\n"
                "   Scoring Anchors:\n"
                f"   {task_info.get('correctness_anchors', '')}\n\n"
                
                "=== SPECIAL INSTRUCTIONS ===\n"
                "If the model output is empty or completely unrelated to the task (e.g., 'Sorry I cannot...'), both scores should be low.\n" 
                "If Format is wrong (e.g. JSON required but got Prose), Intent Score MUST be < 40.\n\n"

                "=== OUTPUT FORMAT (STRICT) ===\n"
                "Rationale: [Brief explanation]\n"
                "Intent: [single number 0-100]\n"
                "Correctness: [single number 1-100]\n\n"
                
                "DO NOT calculate weighted scores. Output ONLY the three lines above."
            )

            # Call MLLM judge with the query image
            judge_response = call_vllm_judge(judge_prompt, image_path)

            try:
                rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Intent:|$)', judge_response, re.IGNORECASE | re.DOTALL)
                if rationale_match:
                    rationale = rationale_match.group(1).strip()

                intent_match = re.search(r'Intent[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                correctness_match = re.search(r'Correctness[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)

                if intent_match:
                    intent_score = float(intent_match.group(1))
                    intent_score = min(100.0, max(0.0, intent_score))  # Allow 0 for intent

                if correctness_match:
                    correctness_score = float(correctness_match.group(1))
                    correctness_score = min(100.0, max(1.0, correctness_score))


                if intent_match and correctness_match:
                    total_intent_score += intent_score
                    total_correctness_score += correctness_score
                    valid_count += 1
            except Exception as e:
                print(f"Error parsing judge response: {e}")


        final_score = intent_score * 0.6 + correctness_score * 0.4

        results.append({
            'image_name': image_name,
            'intent': intent,
            'base_intent': base_intent,
            'intent_suffix': intent_suffix,
            'text': text,  # Keyword instruction
            'num_demos': len(demos),
            'standard_answer': answer,
            'generated_answer': generated_answer,
            'rationale': rationale,
            'intent_score': intent_score,
            'correctness_score': correctness_score,
            'final_score': final_score
        })

    mean_intent = total_intent_score / valid_count if valid_count > 0 else 0.0
    mean_correctness = total_correctness_score / valid_count if valid_count > 0 else 0.0
    mean_overall = mean_intent * 0.6 + mean_correctness * 0.4  # Intent 60%, Correctness 40%

    print(f"\nAnalogical Inference Results:")
    print(f"  Mean Intent Score: {mean_intent:.2f}/100")
    print(f"  Mean Correctness Score: {mean_correctness:.2f}/100")
    print(f"  Mean Overall Score: {mean_overall:.2f}/100")
    print(f"  Valid samples: {valid_count}/{len(data)}")

    result_data = {
        'mean_intent_score': mean_intent,
        'mean_correctness_score': mean_correctness,
        'mean_overall_score': mean_overall,
        'valid_count': valid_count,
        'total_count': len(data),
        'results': results
    }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results


def eval_chain_of_editing(inferencer, data_path, image_dir, output_path, num_demos=0, gen_output_dir=None):
    """Public release documentation."""
    print(f"\n=== Evaluating Chain-of-Editing (Cumulative Preservation) ===")

    with open(data_path, "r") as f:
        data = json.load(f)

    if gen_output_dir is None:
        gen_output_dir = os.path.join(os.path.dirname(output_path), "chain_generated")
    os.makedirs(gen_output_dir, exist_ok=True)

    results = []
    all_sample_scores = []
    skipped = 0

    inference_params = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        do_sample=False,
        text_temperature=0.1,
    )

    for item in tqdm(data, desc="Chain-Edit"):
        sample_id = item.get("id")
        original_image_name = item.get("original_image", f"sample_{sample_id}_0.jpg")
        original_image_path = os.path.join(image_dir, original_image_name)
        edit_steps = item.get("edit_steps", [])

        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image not found: {original_image_path}")

        if len(edit_steps) < 1:
            skipped += 1
            continue


        original_img = Image.open(original_image_path).convert("RGB")
        current_img = original_img
        current_img_path = original_image_path


        sample_results = []
        edit_history = []  # [(instruction, generated_image), ...]


        sample_total_degradation = 0.0
        sample_edit_accuracy_scores = []

        for step_idx, step in enumerate(edit_steps):
            instruction = step.get("instruction", "")


            if step_idx == 0:

                input_list = ["User: ", original_img, instruction, "\nAssistant:"]
            else:

                input_list = ["User: ", original_img]
                for hist_inst, hist_img in edit_history:
                    input_list.append(hist_inst)
                    input_list.append("\nAssistant: ")
                    input_list.append(hist_img)
                input_list.append(f"\nUser: {instruction}\nAssistant:")


            gen_path = os.path.join(gen_output_dir, f"chain_{sample_id}_step{step_idx}.jpg")
            prev_gen_path = current_img_path

            try:
                output_list = inferencer.interleave_inference(
                    input_lists=input_list,
                    understanding_output=False,
                    think=False,
                    **inference_params
                )
                generated_img = output_list[-1]
                if isinstance(generated_img, Image.Image):
                    generated_img.save(gen_path)
                    edit_history.append((instruction, generated_img))


                    step_result = {
                        "step_idx": step_idx,
                        "instruction": instruction,
                        "generated_path": gen_path,
                        "evaluated": False
                    }


                    curr_gt_filename = step.get("reference_image")
                    curr_gt_path = None

                    if curr_gt_filename:
                        curr_gt_path = os.path.join(image_dir, curr_gt_filename)


                    if not curr_gt_path or not os.path.exists(curr_gt_path):
                        raise FileNotFoundError(f"Current GT image not found for sample {sample_id} step {step_idx}: {curr_gt_path}")




                    if step_idx == 0:
                        judge_prompt = (
                            "You are an expert image editing evaluator.\n\n"
                            "**You will see 2 images in order:**\n"
                            "  1. Model Output - The model's generated image after this edit step\n"
                            "  2. Reference (GT) - The ground truth result after this edit step\n\n"
                        )
                        images_to_judge = [gen_path, curr_gt_path]
                    else:
                        judge_prompt = (
                            "You are an expert image editing evaluator.\n\n"
                            "**You will see 3 images in order:**\n"
                            "  1. Previous Output - The image from the previous edit step\n"
                            "  2. Current Output - The model's generated image after this edit step\n"
                            "  3. Reference (GT) - The ground truth result after this edit step\n\n"
                        )
                        images_to_judge = [prev_gen_path, gen_path, curr_gt_path]

                    judge_prompt += (
                        f"**Edit Instruction**: {instruction}\n\n"
                        "Evaluate on TWO dimensions. DO NOT calculate any total score.\n\n"

                        "**1. Edit Accuracy (1-100)** - How well does the current output match the GT?\n"
                        "Compare the current output with the reference GT to evaluate editing quality.\n"
                        "- 85-100: Output matches GT very well, all instruction details correctly implemented\n"
                        "- 70-84: Good match with GT, minor differences\n"
                        "- 50-69: Partial match, noticeable differences from GT\n"
                        "- 25-49: Significant differences from GT\n"
                        "- 1-24: Output very different from GT or incorrectly edited\n\n"

                        "**2. Preservation Degradation (0-100)** - How much have non-target regions degraded?\n"
                        "This is a CUMULATIVE degradation score that adds up across editing steps.\n"
                        "Score the degradation of regions that should NOT have been edited:\n"
                        "- 0: Perfect preservation, no degradation at all\n"
                        "- 1-15: Minor degradation (slight blur, small artifacts, tiny color shifts)\n"
                        "- 16-30: Noticeable degradation (visible quality loss, noticeable artifacts)\n"
                        "- 31-50: Significant degradation (clear quality loss, distortion)\n"
                        "- 51-100: Severe degradation (major corruption, widespread changes)\n\n"
                        "**IMPORTANT**: Preservation Degradation accumulates. Even small degradation in each step adds up.\n\n"

                        "**Output Format**:\n"
                        "Rationale: [Brief assessment]\n"
                        "Edit_Accuracy: [1-100]\n"
                        "Preservation_Degradation: [0-100]"
                    )

                    judge_response = call_vllm_judge(judge_prompt, images_to_judge)

                    try:

                        rationale_match = re.search(r'Rationale[:\s]+(.+?)(?=Edit_Accuracy|$)', judge_response, re.IGNORECASE | re.DOTALL)
                        rationale = rationale_match.group(1).strip() if rationale_match else ""


                        accuracy_match = re.search(r'Edit[_\s]*Accuracy[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
                        degradation_match = re.search(r'Preservation[_\s]*Degradation[:\s]+(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)

                        edit_accuracy = 0.0
                        degradation = 0.0

                        if accuracy_match:
                            edit_accuracy = min(100.0, max(1.0, float(accuracy_match.group(1))))
                            sample_edit_accuracy_scores.append(edit_accuracy)

                        if degradation_match:
                            degradation = min(100.0, max(0.0, float(degradation_match.group(1))))
                            sample_total_degradation += degradation

                        step_result.update({
                            "evaluated": True,
                            "prev_output_path": prev_gen_path,
                            "curr_gt_path": curr_gt_path,
                            "edit_accuracy": edit_accuracy,
                            "preservation_degradation": degradation,
                            "rationale": rationale
                        })
                    except Exception as e:
                        print(f"Error parsing step scores for sample {sample_id} step {step_idx}: {e}")

                    sample_results.append(step_result)


                    current_img = generated_img
                    current_img_path = gen_path
                else:
                    print(f"Failed to generate step {step_idx} for sample {sample_id}")
                    break
            except Exception as e:
                print(f"Error generating step {step_idx} for sample {sample_id}: {e}")
                break


        sample_mean_edit_accuracy = 0.0
        sample_preservation_score = 0.0
        sample_final_score = 0.0

        if len(sample_edit_accuracy_scores) > 0:

            sample_mean_edit_accuracy = sum(sample_edit_accuracy_scores) / len(sample_edit_accuracy_scores)


            sample_preservation_score = max(0.0, 100.0 - sample_total_degradation)


            sample_final_score = sample_mean_edit_accuracy * 0.5 + sample_preservation_score * 0.5


            all_sample_scores.append(sample_final_score)

        results.append({
            "id": sample_id,
            "num_steps": len(sample_results),
            "steps": sample_results,
            "mean_edit_accuracy": sample_mean_edit_accuracy,
            "total_degradation": sample_total_degradation,
            "preservation_score": sample_preservation_score,
            "final_score": sample_final_score
        })


    mean_final_score = sum(all_sample_scores) / len(all_sample_scores) if all_sample_scores else 0.0


    all_edit_accuracies = []
    all_degradations = []
    for r in results:
        if r["final_score"] > 0:
            all_edit_accuracies.append(r["mean_edit_accuracy"])
            all_degradations.append(r["total_degradation"])

    mean_edit_accuracy = sum(all_edit_accuracies) / len(all_edit_accuracies) if all_edit_accuracies else 0.0
    mean_degradation = sum(all_degradations) / len(all_degradations) if all_degradations else 0.0
    mean_preservation = sum([r["preservation_score"] for r in results if r["final_score"] > 0]) / len([r for r in results if r["final_score"] > 0]) if any(r["final_score"] > 0 for r in results) else 0.0

    print(f"\n=== Chain-of-Editing Results ===")
    print(f"Total samples: {len(data)}")
    print(f"Valid samples: {len(all_sample_scores)}")
    print(f"Skipped: {skipped}")
    print(f"\nPer-Sample Statistics (for reference):")
    print(f"  Mean Edit Accuracy:     {mean_edit_accuracy:.2f}/100")
    print(f"  Mean Degradation:       {mean_degradation:.2f}")
    print(f"  Mean Preservation:      {mean_preservation:.2f}/100")
    print(f"\nFinal Score:              {mean_final_score:.2f}/100 (average of per-sample scores)")

    result_data = {
        "final_score": mean_final_score,
        "mean_edit_accuracy": mean_edit_accuracy,
        "mean_degradation": mean_degradation,
        "mean_preservation": mean_preservation,
        "total_samples": len(data),
        "valid_samples": len(all_sample_scores),
        "skipped": skipped,
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return results
