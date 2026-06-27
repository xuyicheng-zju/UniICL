"""Public release module documentation."""
import os
import json
import argparse
import re
import base64
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import sys
import requests
import numpy as np
from scipy.stats import spearmanr, pearsonr
from bert_score import score as bert_score
from safetensors.torch import load_file


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "UniICL"))

from modeling.uniicl import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from models.capm import CapmConfig
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from inferencer import InterleaveInferencer
from public_path_config import (
    DEFAULT_JUDGE_API_BASE,
    DEFAULT_JUDGE_MODEL,
    TASK_DATA_REL_PATHS,
)
from utils.scoring import load_hpsv3_model


CAPM_ABLATION_CHOICES = [
    "none",
    "no_adaptive_routing",
    "no_decoupled_encoding",
    "no_low_rank_transformation",
]


VLLM_API_BASE = DEFAULT_JUDGE_API_BASE
JUDGE_MODEL = DEFAULT_JUDGE_MODEL


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


class SafeInterleaveInferencer(InterleaveInferencer):
    """Public release documentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = next(self.model.language_model.model.embed_tokens.parameters()).device

    def _to_device(self, data):
        """Public release documentation."""
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self._to_device(item) for item in data]
        else:
            return data

    def _parse_understanding_icl_for_capm(self, input_lists):
        """Public release documentation."""
        turns = []
        cur_turn = None

        def _new_turn():
            return {
                "item_indices": [],
                "image": None,
                "user_parts": [],
                "assistant_text": "",
                "has_assistant": False,
                "assistant_open": False,
            }

        def _has_content(turn):
            if turn is None:
                return False
            return (
                len(turn["item_indices"]) > 0
                or turn["image"] is not None
                or any(p.strip() for p in turn["user_parts"])
                or turn["has_assistant"]
            )

        def _finalize_turn():
            nonlocal cur_turn
            if not _has_content(cur_turn):
                cur_turn = None
                return
            user_text = " ".join(
                p.strip() for p in cur_turn["user_parts"] if isinstance(p, str) and p.strip()
            ).strip()
            turns.append({
                "item_indices": list(cur_turn["item_indices"]),
                "image": cur_turn["image"],
                "user_text": user_text,
                "assistant_text": cur_turn["assistant_text"].strip(),
                "has_assistant": cur_turn["has_assistant"],
                "assistant_open": cur_turn["assistant_open"],
            })
            cur_turn = None

        def _append_assistant_text(turn, text):
            if not text:
                return
            if turn["assistant_text"]:
                turn["assistant_text"] += " " + text.strip()
            else:
                turn["assistant_text"] = text.strip()

        for idx, item in enumerate(input_lists):
            if isinstance(item, Image.Image):
                if cur_turn is None:
                    cur_turn = _new_turn()
                cur_turn["item_indices"].append(idx)
                if cur_turn["image"] is None:
                    cur_turn["image"] = item
                continue

            if not isinstance(item, str):
                continue

            text = item
            stripped = text.strip()
            if stripped == "":
                if cur_turn is not None:
                    cur_turn["item_indices"].append(idx)
                continue

            if re.match(r'^\s*User:\s*', text, flags=re.IGNORECASE):
                if _has_content(cur_turn):
                    _finalize_turn()
                cur_turn = _new_turn()
                cur_turn["item_indices"].append(idx)
                payload = re.sub(r'^\s*User:\s*', '', text, flags=re.IGNORECASE)

                split_match = re.search(r'\n\s*Assistant:\s*', payload, flags=re.IGNORECASE)
                if split_match:
                    user_part = payload[:split_match.start()].strip()
                    asst_part = payload[split_match.end():].strip()
                    if user_part:
                        cur_turn["user_parts"].append(user_part)
                    cur_turn["has_assistant"] = True
                    cur_turn["assistant_open"] = (asst_part == "")
                    _append_assistant_text(cur_turn, asst_part)
                else:
                    user_part = payload.strip()
                    if user_part:
                        cur_turn["user_parts"].append(user_part)
                continue

            if re.match(r'^\s*Assistant:\s*', text, flags=re.IGNORECASE):
                if cur_turn is None:
                    cur_turn = _new_turn()
                cur_turn["item_indices"].append(idx)
                payload = re.sub(r'^\s*Assistant:\s*', '', text, flags=re.IGNORECASE).strip()
                cur_turn["has_assistant"] = True
                cur_turn["assistant_open"] = (payload == "")
                _append_assistant_text(cur_turn, payload)
                continue

            if cur_turn is None:
                cur_turn = _new_turn()
            cur_turn["item_indices"].append(idx)
            if cur_turn["has_assistant"]:
                _append_assistant_text(cur_turn, text)
            else:
                cur_turn["user_parts"].append(text)

        if _has_content(cur_turn):
            _finalize_turn()

        if len(turns) < 2:
            return [], set()

        query_turn_idx = None
        for i in range(len(turns) - 1, -1, -1):
            if turns[i]["has_assistant"] and turns[i]["assistant_open"]:
                query_turn_idx = i
                break
        if query_turn_idx is None:
            query_turn_idx = len(turns) - 1

        query_item_indices = set(turns[query_turn_idx]["item_indices"])

        demo_tuples = []
        for turn in turns[:query_turn_idx]:
            if turn["image"] is None:
                continue
            if not turn["has_assistant"]:
                continue
            if turn["assistant_open"]:
                continue
            demo_tuples.append((turn["image"], turn["user_text"], turn["assistant_text"]))

        return demo_tuples, query_item_indices

    @torch.no_grad()
    def _encode_capm_demos(self, demo_tuples):
        """Encode demo tuples into CAPM pattern bank.
        
        Args:
            demo_tuples: List of (PIL.Image, user_text, asst_text) triples
                         user_text: question/instruction text  (segment 0)
                         asst_text: answer/annotation text     (segment 1)
        """
        if not demo_tuples or self.model.capm is None:
            return
        
        from data.data_utils import pil_img2rgb, patchify
        
        demo_embeds_list = []
        segment_ids_list = []
        
        for image, user_text, asst_text in demo_tuples:
            embeds_parts = []
            seg_parts = []
            
            # 1) Image embeddings → segment 0 (User/Observation)
            image_pil = self.vae_transform.resize_transform(pil_img2rgb(image))
            image_tensor = self.vit_transform(image_pil)
            vit_position_ids = self.model.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.model.vit_patch_size,
                max_num_patches_per_side=self.model.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.model.vit_patch_size)
            
            vit_tokens = vit_tokens.to(device=self.device)
            vit_position_ids = vit_position_ids.to(device=self.device)
            cu_seqlens = torch.tensor([0, vit_tokens.shape[0]], dtype=torch.int32, device=self.device)
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                vit_embed = self.model.vit_model(
                    packed_pixel_values=vit_tokens,
                    packed_flattened_position_ids=vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=vit_tokens.shape[0],
                )
                vit_embed = self.model.connector(vit_embed)
                pos_emb = self.model.vit_pos_embed(vit_position_ids)
                vit_embed = (vit_embed + pos_emb).to(torch.bfloat16)
            
            embeds_parts.append(vit_embed)
            seg_parts.append(torch.zeros(vit_embed.shape[0], device=self.device, dtype=torch.long))
            
            # 2) User text → segment 0
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                if user_text:
                    user_ids = self.tokenizer.encode(user_text)
                    user_ids_t = torch.tensor(user_ids, dtype=torch.long, device=self.device)
                    user_embed = self.model.language_model.model.embed_tokens(user_ids_t)
                    embeds_parts.append(user_embed)
                    seg_parts.append(torch.zeros(len(user_ids), device=self.device, dtype=torch.long))
                
                # 3) Assistant text → segment 1
                if asst_text:
                    asst_ids = self.tokenizer.encode(asst_text)
                    asst_ids_t = torch.tensor(asst_ids, dtype=torch.long, device=self.device)
                    asst_embed = self.model.language_model.model.embed_tokens(asst_ids_t)
                    embeds_parts.append(asst_embed)
                    seg_parts.append(torch.ones(len(asst_ids), device=self.device, dtype=torch.long))
            
            # Pack: (1, L_demo, d_backbone)
            demo_embed = torch.cat(embeds_parts, dim=0).unsqueeze(0)
            demo_seg = torch.cat(seg_parts, dim=0).unsqueeze(0)
            
            demo_embeds_list.append(demo_embed)
            segment_ids_list.append(demo_seg)
        
        # Encode all demos → cached pattern bank
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            self.model.capm.encode_demos(demo_embeds_list, segment_ids_list)
        
        print(f"  [CAPM] Encoded {len(demo_tuples)} demos into pattern bank")

    @torch.no_grad()
    def update_context_text(self, text, gen_context, capm_active=False):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._to_device(generation_input)

        past_key_values = self.model.forward_cache_update_text(past_key_values, capm_active=capm_active, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True, capm_active=False):
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._to_device(generation_input)
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model,
                past_key_values,
                capm_active=capm_active,
                **generation_input,
            )

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = self._to_device(generation_input)
            past_key_values = self.model.forward_cache_update_vit(past_key_values, capm_active=capm_active, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0, capm_active=False):
        from copy import deepcopy
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = self._to_device(generation_input)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            capm_active=capm_active,
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=28,
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        """Public release documentation."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        generation_input = self._to_device(generation_input)

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = self._to_device(generation_input_cfg_text)

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = self._to_device(generation_input_cfg_img)

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists,
        think=False,
        understanding_output=False,
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.1,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=28,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        num_demos=0,
    ):
        """Public release documentation."""
        from copy import deepcopy
        from data.data_utils import pil_img2rgb
        from inferencer import VLM_THINK_SYSTEM_PROMPT, GEN_THINK_SYSTEM_PROMPT

        # ── Auto-detect demos & extract CAPM tuples ──
        # Supports two ICL formats:
        #   Format A (utils/icl.py): ["\nUser: ", img, question, "\nAssistant: answer", ...]
        #     → each demo = 4 items: [str, Image, str, str]
        #   Format B (eval_bagel.py old): [img, "User: Q Assistant: A", ...]
        #     → each demo = 2 items: [Image, str]
        demo_tuples = []  # [(Image, user_text, asst_text), ...]
        demo_item_count = 0
        query_item_indices = set()

        if self.model.capm is not None and understanding_output:
            parsed_demos, parsed_query_item_indices = self._parse_understanding_icl_for_capm(input_lists)
            demo_tuples = parsed_demos
            query_item_indices = parsed_query_item_indices
            if demo_tuples:
                num_demos = len(demo_tuples)

        if self.model.capm is not None and num_demos == 0 and not demo_tuples:
            # Try Format A first: [str, Image, str, str] per demo
            # Target query also has same format, so last group = target
            i = 0
            detected_a = []
            while i + 3 < len(input_lists):
                if (isinstance(input_lists[i], str) and 
                    isinstance(input_lists[i+1], Image.Image) and
                    isinstance(input_lists[i+2], str) and
                    isinstance(input_lists[i+3], str) and
                    'Assistant' in input_lists[i+3]):
                    # Extract: user_prefix + question = user_text, answer = asst_text
                    user_text = input_lists[i].strip() + " " + input_lists[i+2].strip()
                    asst_text = input_lists[i+3].strip()
                    # Clean "Assistant:" prefix from asst_text for segment
                    if asst_text.startswith("\nAssistant:"):
                        asst_text = asst_text[len("\nAssistant:"):].strip()
                    elif asst_text.startswith("Assistant:"):
                        asst_text = asst_text[len("Assistant:"):].strip()
                    detected_a.append((input_lists[i+1], user_text, asst_text))
                    i += 4
                else:
                    break
            
            if len(detected_a) >= 2:
                # Last group is target query, all preceding groups are demos
                demo_tuples = detected_a[:-1]
                demo_item_count = len(demo_tuples) * 4
                num_demos = len(demo_tuples)
                query_item_indices = set(range(demo_item_count, len(input_lists)))
            
            # Fallback: Try Format B: [Image, str] per demo
            if not demo_tuples:
                i = 0
                detected_b = []
                while i + 1 < len(input_lists):
                    if (isinstance(input_lists[i], Image.Image) and 
                        isinstance(input_lists[i+1], str)):
                        detected_b.append(i)
                        i += 2
                    else:
                        break
                if len(detected_b) >= 2:  # at least 1 demo + 1 target
                    for idx in detected_b[:-1]:  # exclude last (target)
                        text = input_lists[idx + 1]
                        if "Assistant:" in text:
                            parts = text.split("Assistant:", 1)
                            user_text = parts[0].strip()
                            asst_text = parts[1].strip()
                        else:
                            user_text = text
                            asst_text = ""
                        demo_tuples.append((input_lists[idx], user_text, asst_text))
                    demo_item_count = len(demo_tuples) * 2
                    num_demos = len(demo_tuples)
                    query_item_indices = set(range(demo_item_count, len(input_lists)))

        use_capm = (len(demo_tuples) > 0 and self.model.capm is not None)
        full_gate_mode = use_capm and bool(
            getattr(getattr(self.model.capm, "config", None), "apply_to_demo_tokens", False)
        )

        # Encode demos into CAPM pattern bank
        if use_capm:
            self._encode_capm_demos(demo_tuples)
            capm_route_count = 0  # stats counter

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for item_idx, input_term in enumerate(input_lists):
                # CAPM gate scope:
                # - default: query-only
                # - full mode: all context items (demo + query)
                if full_gate_mode:
                    capm_active = use_capm
                elif query_item_indices:
                    capm_active = use_capm and (item_idx in query_item_indices)
                else:
                    capm_active = use_capm and (item_idx >= demo_item_count)
                if capm_active:
                    capm_route_count += 1

                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context, capm_active=capm_active)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output, capm_active=capm_active)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n, capm_active=use_capm)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n, capm_active=use_capm)
                    gen_context = self.update_context_text(gen_text, gen_context, capm_active=use_capm)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        # Clear CAPM cache after inference & print stats
        if use_capm and self.model.capm is not None:
            tau_base = getattr(self.model.capm.aligner, 'tau_base', None)
            tau_base_str = f"{tau_base.item():.4f}" if tau_base is not None else "N/A"
            if full_gate_mode:
                scope_text = "all items"
                routed_count = capm_route_count
            else:
                scope_text = "query items"
                routed_count = len(query_item_indices) if query_item_indices else capm_route_count
            print(
                f"  [CAPM] {num_demos} demos | {routed_count} {scope_text} gated | "
                f"route_calls={capm_route_count} | tau_base={tau_base_str}"
            )
            self.model.capm.clear_cache()

        return output_list


def load_bagel_model(model_path, base_model_path=None, use_mixed_weights=False, enable_offload=False, enable_capm=True):
    """Public release documentation."""

    def _remap_legacy_prism_keys(state_dict):
        """Map legacy PRISM checkpoint keys onto the renamed CAPM module."""
        prism_keys = [k for k in state_dict.keys() if k.startswith("prism.")]
        if not prism_keys:
            return state_dict, 0, 0

        remapped_state_dict = {}
        remapped_count = 0
        skipped_count = 0

        for key, value in state_dict.items():
            if key.startswith("prism."):
                remapped_key = "capm." + key[len("prism."):]
                if remapped_key in state_dict:
                    skipped_count += 1
                    continue
                remapped_state_dict[remapped_key] = value
                remapped_count += 1
            else:
                remapped_state_dict[key] = value

        return remapped_state_dict, remapped_count, skipped_count

    def _load_checkpoint_state_dict(checkpoint_path):
        state_dict = load_file(checkpoint_path)
        state_dict, remapped_count, skipped_count = _remap_legacy_prism_keys(state_dict)
        if remapped_count > 0:
            print(
                f"  Remapped {remapped_count} legacy PRISM keys to CAPM keys"
                + (f" ({skipped_count} skipped due to existing capm.* keys)" if skipped_count > 0 else "")
            )
        return state_dict

    config_path = base_model_path if (use_mixed_weights and base_model_path) else model_path

    print(f"Loading model configs from {config_path}...")
    if use_mixed_weights:
        print(f"  [Mixed Weights Mode] Base: {base_model_path}, Finetuned: {model_path}")

    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(config_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(config_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1


    vae_model, vae_config = load_ae(local_path=os.path.join(config_path, "ae.safetensors"))
    vae_model = vae_model.to(device='cuda', dtype=torch.bfloat16)
    vae_model.eval()


    capm_config = None
    capm_config_path = os.path.join(model_path, "capm_config.json")
    if not enable_capm:
        print("⏭️ CAPM disabled by --no-capm flag, skipping CAPM config loading")
    elif os.path.exists(capm_config_path):
        print(f"Loading CAPM config from {capm_config_path}...")
        with open(capm_config_path, 'r') as f:
            capm_dict = json.load(f)
        capm_config = CapmConfig(**capm_dict)
        print(f"✅ CAPM config loaded: d_capm={capm_config.d_capm}, num_inject_layers={capm_config.num_inject_layers}")
    else:
        print(f"⚠️ No capm_config.json found at {capm_config_path}, CAPM disabled")

    # Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        capm_config=capm_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )


    tokenizer = Qwen2Tokenizer.from_pretrained(config_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)


    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=False)

    # ========================================

    # ========================================
    if use_mixed_weights and base_model_path:



        base_checkpoint = os.path.join(base_model_path, "ema.safetensors")
        if not os.path.exists(base_checkpoint):
            base_checkpoint = os.path.join(base_model_path, "model.safetensors")

        print(f"Step 1: Loading BASE weights from {base_checkpoint}...")
        base_state_dict = _load_checkpoint_state_dict(base_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(base_state_dict, strict=False)
        print(f"  Base weights loaded: {len(base_state_dict)} keys")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        del base_state_dict


        finetuned_checkpoint = os.path.join(model_path, "ema.safetensors")
        if not os.path.exists(finetuned_checkpoint):
            finetuned_checkpoint = os.path.join(model_path, "model.safetensors")

        print(f"Step 2: Loading FINETUNED weights from {finetuned_checkpoint}...")
        finetuned_state_dict = _load_checkpoint_state_dict(finetuned_checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(finetuned_state_dict, strict=False)
        print(f"  Finetuned weights loaded: {len(finetuned_state_dict)} keys (overwriting Und parts)")
        if missing_keys:
            print(f"  Missing keys after finetuned load: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys after finetuned load: {len(unexpected_keys)}")
        del finetuned_state_dict

    else:

        checkpoint_path = os.path.join(model_path, "ema.safetensors")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(model_path, "model.safetensors")

        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = _load_checkpoint_state_dict(checkpoint_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        del state_dict

    print("Moving model to CUDA...")
    model = model.to(device='cuda', dtype=torch.bfloat16)
    model.eval()

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def build_icl_input(demos, image_dir, target_image_path, target_question):
    """Public release documentation."""
    input_list = []


    for demo in demos:
        demo_img_path = os.path.join(image_dir, demo['image_name'])
        if os.path.exists(demo_img_path):
            demo_img = Image.open(demo_img_path).convert("RGB")
            input_list.append(demo_img)




            demo_question = demo.get('instruction', demo.get('text', ''))
            demo_text = f"User: {demo_question} Assistant: {demo.get('answer', demo.get('annotation', ''))}"
            input_list.append(demo_text)


    target_img = Image.open(target_image_path).convert("RGB")
    input_list.append(target_img)
    input_list.append(f"User: {target_question} Assistant: ")

    return input_list


def compute_iou(box1, box2):
    """Public release documentation."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])


    intersection = max(0, x2 - x1) * max(0, y2 - y1)


    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])


    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def parse_bbox(text):
    """Public release documentation."""
    import re

    pattern = r'\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?'
    match = re.search(pattern, text)
    if match:
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            return coords
        except ValueError:
            return None
    return None


def normalize_bbox(bbox, image_width, image_height):
    """Public release documentation."""
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox


    if max(x1, y1, x2, y2) > 1.0:
        x1 = x1 / image_width
        y1 = y1 / image_height
        x2 = x2 / image_width
        y2 = y2 / image_height


    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    return [x1, y1, x2, y2]


def parse_option_label(text, valid_labels, strict_action=False):
    """Public release documentation."""
    if not text:
        return None

    upper_text = text.strip().upper()
    

    action_match = re.search(r'<action>\s*([A-Z])\s*</action>', text, re.IGNORECASE)
    if action_match:
        letter = action_match.group(1).upper()
        if letter in valid_labels:
            return letter
            
    if strict_action:
        return None


    num_match = re.search(r'\b(\d+)\b', upper_text)
    if num_match:
        candidate = num_match.group(1)
        if candidate in valid_labels:
            return candidate


    letter_match = re.search(r'\b([A-Z])\b', upper_text)
    if letter_match:
        letter = letter_match.group(1)
        if letter in valid_labels:
            return letter
    return None



# Import evaluation functions from utils
from utils.evaluators import (
    eval_grounding,
    eval_attr_rec_gen,
    eval_vqa_gen,
    eval_caption_styled,
    eval_t2i,
    eval_i2i_editing,
    eval_aesthetic_assessment,
    eval_authenticity_detection,
    eval_image_perfection,
    eval_fcb_classification,
    eval_fci_t2i,
    eval_planning,
    eval_visualcloze_g,
    eval_visualcloze_u,
    eval_chain_of_editing,
)
from utils.judge import set_judge_config
from public_path_config import (
    DEFAULT_BAGEL_BASE_MODEL,
    DEFAULT_HPSV3_CHECKPOINT,
    DEFAULT_HPSV3_CONFIG,
    DEFAULT_JUDGE_API_BASE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_QALIGN_MODEL,
    CANONICAL_TASK_ORDER,
    normalize_task_name,
    TASK_DATA_REL_PATHS,
)


def main():
    parser = argparse.ArgumentParser(description="UniICL-Bench Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to UniICL model (finetuned checkpoint)")
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BAGEL_BASE_MODEL,
                       help="Path to base UniICL model (for mixed weights mode)")
    parser.add_argument("--use-mixed-weights", action="store_true",
                       help="Use mixed weights: load base Gen weights + finetuned Und weights")
    parser.add_argument("--task", type=str,
                       default="all",
                       help=f"Task to evaluate using paper-aligned snake_case names (choices: {', '.join(CANONICAL_TASK_ORDER)}, all)")
    parser.add_argument("--data-path", type=str,
                       help="Path to task JSONL file (required if task is not 'all')")
    parser.add_argument("--image-dir", type=str,
                       help="Directory containing images (required if task is not 'all')")
    parser.add_argument("--benchmark-dir", type=str, default=".",
                       help="Root directory of benchmark (for 'all' task mode)")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--k-shot", type=int, default=0,
                       help="K-shot for ICL demonstrations (default: 0)")
    parser.add_argument("--skip-visual-refinement-scoring",
                       dest="skip_visual_refinement_scoring", action="store_true",
                       help="Skip Q-Align scoring for visual_refinement (only generate images)")
    parser.add_argument("--skip-perfection-scoring",
                       dest="skip_visual_refinement_scoring", action="store_true",
                       help=argparse.SUPPRESS)
    parser.add_argument("--enable-offload", action="store_true",
                       help="Enable model offload (slower, less VRAM)")
    parser.add_argument("--no-capm", action="store_true",
                       help="Disable CAPM module loading (use base model without CAPM)")
    parser.add_argument("--capm-inject-layers", type=int, default=None,
                       help="Override number of CAPM injection layers (default: use all trained gates, e.g. 28). "
                            "Only the LAST N gates are kept so they match the last N backbone layers. "
                            "Typical ablation values: 28 (full), 14 (top-half), 7 (top-quarter)")
    parser.add_argument("--capm-ablation-mode", type=str, default="none",
                       choices=CAPM_ABLATION_CHOICES,
                       help="Inference-time CAPM component ablation mode. Default keeps original CAPM behavior.")
    parser.add_argument("--capm-fixed-tau", type=float, default=None,
                       help="Optional fixed routing temperature tau for CAPM. Overrides adaptive tau when set.")
    parser.add_argument("--hps-checkpoint", type=str,
                       default=DEFAULT_HPSV3_CHECKPOINT,
                       help="Path to HPSv3 model checkpoint")
    parser.add_argument("--judge-api-base", type=str, default=DEFAULT_JUDGE_API_BASE,
                       help="vLLM API base URL for GPT-Score evaluation")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                       help="Model name for GPT-Score judge")
    parser.add_argument("--analogical-editing-eval-mode", dest="analogical_editing_eval_mode", type=str, default="dinov3",
                       choices=["dinov3", "mllm"],
                       help="Evaluation mode for analogical_editing: dinov3 similarity or mllm judge")
    parser.add_argument("--visualcloze-g-eval-mode", dest="analogical_editing_eval_mode", type=str,
                       choices=["dinov3", "mllm"], help=argparse.SUPPRESS)

    args = parser.parse_args()
    try:
        args.task = normalize_task_name(args.task, allow_all=True)
    except ValueError as e:
        parser.error(str(e))


    set_judge_config(api_base=args.judge_api_base, model=args.judge_model)


    TASK_DATA_MAP = TASK_DATA_REL_PATHS




    TASK_CONFIG = {
        "visual_grounding": (eval_grounding, None),
        "attribute_recognition": (eval_attr_rec_gen, None),
        "scene_reasoning": (eval_vqa_gen, None),
        "style_aware_caption": (eval_caption_styled, None),
        "instructional_generation": (eval_t2i, "hps"),
        "image_manipulation": (eval_i2i_editing, None),
        "aesthetic_assessment": (eval_aesthetic_assessment, None),
        "forgery_detection": (eval_authenticity_detection, None),
        "visual_refinement": (eval_image_perfection, "hps"),
        "fast_concept_mapping": (eval_fcb_classification, None),
        "fast_concept_generation": (eval_fci_t2i, None),
        "world_aware_planning": (eval_planning, None),
        "chain_of_editing": (eval_chain_of_editing, None),
        "analogical_editing": (eval_visualcloze_g, None),
        "analogical_inference": (eval_visualcloze_u, None),
    }


    if args.task and args.task != "all" and not args.data_path:
        args.data_path = os.path.join(args.benchmark_dir, TASK_DATA_MAP[args.task])


    if not args.image_dir:
        parser.error("--image-dir is required. Use run_eval.sh for automatic path resolution.")


    if args.k_shot < 0:
        parser.error(f"--k-shot must be non-negative, got {args.k_shot}")


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
        args.model_path,
        base_model_path=args.base_model_path,
        use_mixed_weights=args.use_mixed_weights,
        enable_offload=args.enable_offload,
        enable_capm=not args.no_capm,
    )

    if model.capm is not None:
        model.capm.config.ablation_mode = args.capm_ablation_mode
        if args.capm_ablation_mode != "none":
            print(f"CAPM ablation mode: {args.capm_ablation_mode}")
        if args.capm_fixed_tau is not None:
            if args.capm_fixed_tau <= 0:
                parser.error(f"--capm-fixed-tau must be positive, got {args.capm_fixed_tau}")
            model.capm.config.fixed_tau = args.capm_fixed_tau
            print(f"CAPM fixed tau: {args.capm_fixed_tau}")
    elif args.capm_ablation_mode != "none":
        print(f"CAPM ablation mode '{args.capm_ablation_mode}' requested, but CAPM is disabled or unavailable")
    elif args.capm_fixed_tau is not None:
        print(f"CAPM fixed tau '{args.capm_fixed_tau}' requested, but CAPM is disabled or unavailable")


    if model.capm is not None and args.capm_inject_layers is not None:
        trained_gates = len(model.capm.gates)
        k = args.capm_inject_layers
        if k <= 0:

            model.capm = None
            print(f"⏭️ --capm-inject-layers=0, CAPM disabled")
        elif k > trained_gates:
            print(f"⚠️ --capm-inject-layers={k} > trained gates ({trained_gates}), using all {trained_gates}")
        elif k < trained_gates:

            import torch.nn as nn
            model.capm.gates = nn.ModuleList(list(model.capm.gates)[-k:])
            print(f"✂️ CAPM gates truncated: {trained_gates} → {k} (injecting last {k} backbone layers)")


    inferencer = SafeInterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )


    hps_model = None
    if args.task in ["instructional_generation", "all"]:
        print(f"\nLoading HPSv3 model from {args.hps_checkpoint}...")
        hps_model = load_hpsv3_model(args.hps_checkpoint)
        if hps_model:
            print("✅ HPSv3 model loaded successfully")
        else:
            print("⚠️ HPSv3 model not loaded, Instructional Generation scoring will be skipped")


    qalign_model = None
    if args.task in ["visual_refinement", "all"]:
        print("\nLoading Q-Align model for Visual Refinement evaluation...")
        from utils.scoring import load_qalign_model
        try:
            qalign_model = load_qalign_model(DEFAULT_QALIGN_MODEL)
            print("✅ Q-Align model loaded successfully")
        except Exception as e:
            print(f"⚠️ Q-Align model not loaded: {e}")


    if args.task == "all":

        parser.error("--task all is not supported. Use run_eval.sh to run all tasks with correct image directories.")
    else:

        output_path = os.path.join(output_dir, f"{args.task}_results.json")


        if args.task in TASK_CONFIG:
            eval_func, model_type = TASK_CONFIG[args.task]


            if args.task == "visual_refinement":
                eval_func(inferencer, args.data_path, args.image_dir, output_path, args.k_shot,
                         qalign_model=qalign_model, skip_scoring=args.skip_visual_refinement_scoring)
            elif args.task == "analogical_editing":
                eval_func(
                    inferencer,
                    args.data_path,
                    args.image_dir,
                    output_path,
                    args.k_shot,
                    eval_mode=args.analogical_editing_eval_mode,
                )
            elif model_type == "hps":
                eval_func(inferencer, args.data_path, args.image_dir, output_path, args.k_shot, hps_model=hps_model)
            else:
                eval_func(inferencer, args.data_path, args.image_dir, output_path, args.k_shot)
        else:
            raise ValueError(f"Unknown task: {args.task}")

    print(f"\n✅ Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
