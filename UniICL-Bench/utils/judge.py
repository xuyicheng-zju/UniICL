"""Public release module documentation."""
import base64
import os
import requests

from public_path_config import DEFAULT_JUDGE_API_BASE, DEFAULT_JUDGE_MODEL


VLLM_API_BASE = DEFAULT_JUDGE_API_BASE
JUDGE_MODEL = DEFAULT_JUDGE_MODEL


def set_judge_config(api_base: str = None, model: str = None):
    """Public release documentation."""
    global VLLM_API_BASE, JUDGE_MODEL
    if api_base:
        VLLM_API_BASE = api_base
    if model:
        JUDGE_MODEL = model


def call_vllm_judge(prompt: str, image_path=None, max_tokens: int = 1024) -> str:
    """Public release documentation."""
    messages = [{"role": "user", "content": []}]


    image_paths = []
    if isinstance(image_path, str):
        image_paths = [image_path]
    elif isinstance(image_path, (list, tuple)):
        image_paths = list(image_path)

    for img in image_paths:
        if img and os.path.exists(img):
            with open(img, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })


    messages[0]["content"].append({"type": "text", "text": prompt})

    try:
        response = requests.post(
            f"{VLLM_API_BASE}/chat/completions",
            json={
                "model": JUDGE_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"vLLM API error: {e}")
        return ""


def mllm_assisted_extraction(prediction_text, question, expected_format, options=None, options_dict=None):
    """Public release documentation."""
    prompt = f"""Given the following question and model prediction, extract the answer in the expected format.

Question: {question}

Expected Format: {expected_format}
"""

    if options:
        prompt += f"\nValid Options: {', '.join(options)}"
    if options_dict:
        prompt += f"\nOptions Dictionary: {options_dict}"

    prompt += f"""

Model Prediction:
{prediction_text}

Please extract ONLY the answer in the expected format. Do not provide explanations.
Answer:"""

    return call_vllm_judge(prompt).strip()
