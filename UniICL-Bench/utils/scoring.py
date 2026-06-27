"""Public release module documentation."""
import sys

import torch
from transformers import AutoModelForCausalLM
from PIL import Image

from public_path_config import (
    DEFAULT_HPSV3_CONFIG,
    DEFAULT_HPSV3_VENDOR_ROOT,
    DEFAULT_QALIGN_MODEL,
)


if DEFAULT_HPSV3_VENDOR_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_HPSV3_VENDOR_ROOT)


# ==================== Q-Align ====================

def load_qalign_model(model_path=None, device="cuda"):
    """Public release documentation."""
    model_path = model_path or DEFAULT_QALIGN_MODEL
    print(f"Loading Q-Align model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    print("Q-Align model loaded successfully!")
    return model


def compute_qalign_score(image_path, qalign_model):
    """Public release documentation."""
    try:

        img = Image.open(image_path).convert('RGB')
        images = [img]


        quality_scores = qalign_model.score(images, task_="quality", input_="image")
        quality_score = float(quality_scores[0])


        aesthetics_scores = qalign_model.score(images, task_="aesthetics", input_="image")
        aesthetics_score = float(aesthetics_scores[0])


        total_score = 0.5 * quality_score + 0.5 * aesthetics_score

        return {
            'quality_score': quality_score,
            'aesthetics_score': aesthetics_score,
            'total_score': total_score
        }
    except Exception as e:
        print(f"Error computing Q-Align score for {image_path}: {e}")
        return None


def compute_qalign_batch_scores(image_paths, qalign_model, batch_size=32):
    """Public release documentation."""
    results = {}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]


        images = []
        valid_paths = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        if not images:
            continue

        try:

            quality_scores = qalign_model.score(images, task_="quality", input_="image")

            aesthetics_scores = qalign_model.score(images, task_="aesthetics", input_="image")


            for path, quality, aesthetics in zip(valid_paths, quality_scores, aesthetics_scores):
                total_score = 0.5 * float(quality) + 0.5 * float(aesthetics)
                results[path] = {
                    'quality_score': float(quality),
                    'aesthetics_score': float(aesthetics),
                    'total_score': total_score
                }
        except Exception as e:
            print(f"Error computing scores for batch: {e}")
            continue

    return results


# ==================== HPSv3 ====================

def load_hpsv3_model(checkpoint_path: str):
    """Public release documentation."""
    try:
        from hpsv3 import HPSv3RewardInferencer

        config_path = DEFAULT_HPSV3_CONFIG
        inferencer = HPSv3RewardInferencer(
            device='cuda',
            config_path=config_path,
            checkpoint_path=checkpoint_path
        )
        return inferencer
    except Exception as e:
        print(f"Error loading HPSv3 model: {e}")
        return None


def compute_hpsv3_score(image_path: str, prompt: str, hps_model) -> float:
    """Public release documentation."""
    if hps_model is None:
        return -1.0
    try:
        reward = hps_model.reward(prompts=[prompt], image_paths=[image_path])
        return float(reward[0][0].item())
    except Exception as e:
        print(f"HPSv3 scoring error: {e}")
        return -1.0


# ==================== CLIP-Score ====================

_clip_model = None
_clip_processor = None


def load_clip_model(model_name="openai/clip-vit-large-patch14", device="cuda"):
    """Public release documentation."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    print(f"Loading CLIP model from {model_name}...")
    from transformers import CLIPProcessor, CLIPModel

    _clip_model = CLIPModel.from_pretrained(model_name).to(device)
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model.eval()
    print("CLIP model loaded successfully!")
    return _clip_model, _clip_processor


def compute_clip_score(image_path: str, prompt: str, clip_model=None, clip_processor=None, device="cuda") -> float:
    """Public release documentation."""
    try:
        if clip_model is None or clip_processor is None:
            clip_model, clip_processor = load_clip_model(device=device)


        image = Image.open(image_path).convert("RGB")


        inputs = clip_processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)


        with torch.no_grad():
            outputs = clip_model(**inputs)

            logits = outputs.logits_per_image


            similarity = logits.squeeze().item()


            score = min(100.0, max(0.0, similarity))

        return score
    except Exception as e:
        print(f"CLIP-Score computation error for {image_path}: {e}")
        return -1.0


def compute_clip_score_batch(image_paths: list, prompts: list, clip_model=None, clip_processor=None, device="cuda", batch_size=32) -> list:
    """Public release documentation."""
    if clip_model is None or clip_processor is None:
        clip_model, clip_processor = load_clip_model(device=device)

    scores = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]

        batch_images = []
        batch_texts = []
        valid_indices = []

        for j, (img_path, prompt) in enumerate(zip(batch_paths, batch_prompts)):
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_texts.append(prompt)
                valid_indices.append(j)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                scores.append(-1.0)

        if not batch_images:
            continue

        try:
            inputs = clip_processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)

            with torch.no_grad():
                outputs = clip_model(**inputs)

                logits = outputs.logits_per_image.diag()
                for score in logits:
                    scores.append(min(100.0, max(0.0, score.item())))
        except Exception as e:
            print(f"Error computing batch CLIP scores: {e}")
            scores.extend([-1.0] * len(batch_images))

    return scores
