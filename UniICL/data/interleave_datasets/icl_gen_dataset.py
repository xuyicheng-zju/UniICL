# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Public release module documentation."""

import io
import os
import random
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def get_instruction(instruction_item):
    """Public release documentation."""
    if isinstance(instruction_item, list):
        return random.choice(instruction_item) if instruction_item else ""
    return instruction_item


class ICLGenIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """Public release documentation."""

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        max_num_tokens_per_sample=36864,
        **kwargs
    ):
        super().__init__(
            dataset_name, transform, tokenizer, vit_transform, 
            data_dir_list, num_used_data, parquet_info,
            local_rank, world_size, num_workers, data_status
        )
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self._debug_truncate = os.environ.get("ICL_DEBUG_TRUNCATE", "0") == "1"
        self._debug_truncate_max = int(os.environ.get("ICL_DEBUG_TRUNCATE_MAX", "20"))

    def _maybe_log_t2i_zero_shot(self, query_data, demo_data):
        if not self._debug_truncate or self._debug_truncate_max <= 0:
            return
        self._debug_truncate_max -= 1

        def _text_tokens(data):
            return sum(len(x) for x in data['text_ids_list'])

        q_text = _text_tokens(query_data)
        d_text = _text_tokens(demo_data)
        q_img = query_data['num_tokens'] - q_text
        d_img = demo_data['num_tokens'] - d_text
        q_shapes = [tuple(t.shape[-2:]) for t in query_data['image_tensor_list']]
        d_shapes = [tuple(t.shape[-2:]) for t in demo_data['image_tensor_list']]
        print(
            "[DEBUG T2I 0-shot] "
            f"query_text={q_text} query_img={q_img} "
            f"demo_text={d_text} demo_img={d_img} "
            f"query_imgs={q_shapes} demo_imgs={d_shapes}"
        )

    def _merge_data(self, target, source):
        target['sequence_plan'].extend(source['sequence_plan'])
        target['text_ids_list'].extend(source['text_ids_list'])
        target['image_tensor_list'].extend(source['image_tensor_list'])
        target['num_tokens'] += source['num_tokens']

    def parse_row(self, row):
        """Public release documentation."""
        image_num = len(row["image_list"])
        instruction_num = len(row["instruction_list"])


        if "num_inputs" in row and row["num_inputs"] is not None:
            return self._parse_visualcloze_g_icl(row, instruction_num)


        if image_num == instruction_num:
            return self._parse_t2i_icl(row, image_num)
        elif image_num == instruction_num * 2:
            return self._parse_i2i_icl(row, instruction_num)
        else:

            for num_inputs in [2, 3, 4]:
                if image_num == instruction_num * (num_inputs + 1):
                    row_with_num_inputs = dict(row)
                    row_with_num_inputs["num_inputs"] = num_inputs
                    return self._parse_visualcloze_g_icl(row_with_num_inputs, instruction_num)

            raise ValueError(f"Invalid ICL format: {image_num} images, {instruction_num} instructions")

    def _parse_t2i_icl(self, row, num_samples):
        """Public release documentation."""
        # 1. Process Query first to check budget
        query_data = self._init_data()
        query_prompt = get_instruction(row["instruction_list"][-1])

        query_data = self._add_text(query_data, "User:", need_loss=False, enable_cfg=False, is_demo=False, segment_id=0)
        # Query Prompt (Condition) -> Enable CFG
        query_data = self._add_text(query_data, query_prompt, need_loss=False, enable_cfg=False, is_demo=False, segment_id=0)

        query_data = self._add_text(query_data, "Assistant:", need_loss=False, enable_cfg=False, is_demo=False, segment_id=1)


        query_data = self._add_image(
            query_data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-1]))),
            need_loss=True,
            need_vae=False,
            need_vit=False,
            enable_cfg=False,
            is_demo=False,
            segment_id=1
        )

        remaining_budget = self.max_num_tokens_per_sample - query_data['num_tokens']
        valid_demos = []
        
        # Original shot count (total samples - 1 query)
        original_shots = num_samples - 1

        # 2. Process Demos from last to first
        for idx in range(num_samples - 2, -1, -1):
            demo_data = self._init_data()
            prompt = get_instruction(row["instruction_list"][idx])
            demo_turn_id = idx + 1  # 1-based demo index for CAPM

            demo_data = self._add_text(demo_data, "User:", need_loss=False, enable_cfg=False, is_demo=True, segment_id=0, demo_turn_id=demo_turn_id)

            demo_data = self._add_text(demo_data, prompt, need_loss=False, enable_cfg=False, is_demo=True, segment_id=0, demo_turn_id=demo_turn_id)

            demo_data = self._add_text(demo_data, "Assistant:", need_loss=False, enable_cfg=False, is_demo=True, segment_id=1, demo_turn_id=demo_turn_id)

            demo_data = self._add_image(
                demo_data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
                is_demo=True,
                segment_id=1,
                demo_turn_id=demo_turn_id
            )

            if demo_data['num_tokens'] <= remaining_budget:
                valid_demos.insert(0, demo_data)
                remaining_budget -= demo_data['num_tokens']
            else:
                # Log truncation info
                kept_shots = len(valid_demos)
                if kept_shots == 0:
                    print(f"[DEBUG T2I 0-shot] Query Cost: {query_data['num_tokens']}, Budget: {self.max_num_tokens_per_sample}, First Demo Cost: {demo_data['num_tokens']}")
                    self._maybe_log_t2i_zero_shot(query_data, demo_data)
                print(f"[Truncate T2I] {self.dataset_name}: {original_shots}-shot -> {kept_shots}-shot. Budget: {self.max_num_tokens_per_sample}")
                break

        # 3. Merge
        final_data = self._init_data()
        for demo in valid_demos:
            self._merge_data(final_data, demo)
        self._merge_data(final_data, query_data)
        return final_data

    def _parse_i2i_icl(self, row, num_samples):
        """Public release documentation."""
        # 1. Process Query
        query_data = self._init_data()
        

        query_data = self._add_text(query_data, "User:", need_loss=False, enable_cfg=False, is_demo=False, segment_id=0)

        # Query source image（condition）-> Enable CFG
        query_data = self._add_image(
            query_data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-2]))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
            enable_cfg=False,
            is_demo=False,
            segment_id=0
        )

        # Query instruction -> Enable CFG
        query_instruction = get_instruction(row["instruction_list"][-1])
        query_data = self._add_text(query_data, query_instruction, need_loss=False, enable_cfg=False, is_demo=False, segment_id=0)


        query_data = self._add_text(query_data, "Assistant:", need_loss=False, enable_cfg=False, is_demo=False, segment_id=1)


        query_data = self._add_image(
            query_data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-1]))),
            need_loss=True,
            need_vae=False,
            need_vit=False,
            enable_cfg=False,
            is_demo=False,
            segment_id=1
        )

        remaining_budget = self.max_num_tokens_per_sample - query_data['num_tokens']
        valid_demos = []
        
        # Original shot count (total samples - 1 query)
        original_shots = num_samples - 1

        # 2. Process Demos from last to first
        for idx in range(num_samples - 2, -1, -1):
            demo_data = self._init_data()
            demo_turn_id = idx + 1  # 1-based demo index for CAPM
            

            demo_data = self._add_text(demo_data, "User:", need_loss=False, enable_cfg=False, is_demo=True, segment_id=0, demo_turn_id=demo_turn_id)


            demo_data = self._add_image(
                demo_data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx * 2]))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
                is_demo=True,
                segment_id=0,
                demo_turn_id=demo_turn_id
            )


            instruction = get_instruction(row["instruction_list"][idx])
            demo_data = self._add_text(demo_data, instruction, need_loss=False, enable_cfg=False, is_demo=True, segment_id=0, demo_turn_id=demo_turn_id)


            demo_data = self._add_text(demo_data, "Assistant:", need_loss=False, enable_cfg=False, is_demo=True, segment_id=1, demo_turn_id=demo_turn_id)


            demo_data = self._add_image(
                demo_data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx * 2 + 1]))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
                is_demo=True,
                segment_id=1,
                demo_turn_id=demo_turn_id
            )

            if demo_data['num_tokens'] <= remaining_budget:
                valid_demos.insert(0, demo_data)
                remaining_budget -= demo_data['num_tokens']
            else:
                # Log truncation info
                kept_shots = len(valid_demos)
                if kept_shots == 0:
                    print(f"[DEBUG I2I 0-shot] Query Cost: {query_data['num_tokens']}, Budget: {self.max_num_tokens_per_sample}, First Demo Cost: {demo_data['num_tokens']}")
                print(f"[Truncate I2I] {self.dataset_name}: {original_shots}-shot -> {kept_shots}-shot. Budget: {self.max_num_tokens_per_sample}")
                break

        # 3. Merge
        final_data = self._init_data()
        for demo in valid_demos:
            self._merge_data(final_data, demo)
        self._merge_data(final_data, query_data)
        return final_data

    def _parse_visualcloze_g_icl(self, row, num_samples):
        """Public release documentation."""
        num_inputs = row["num_inputs"]
        images_per_sample = num_inputs + 1

        # 1. Process Query
        query_base_idx = (num_samples - 1) * images_per_sample
        query_data = self._init_data()


        query_data = self._add_text(query_data, "User:", need_loss=False, enable_cfg=False, is_demo=False, segment_id=0)


        for input_idx in range(num_inputs):
            img_idx = query_base_idx + input_idx
            query_data = self._add_image(
                query_data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][img_idx]))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
                is_demo=False,
                segment_id=0
            )

        # Query instruction -> Enable CFG
        query_instruction = get_instruction(row["instruction_list"][-1])
        if query_instruction and query_instruction.strip():
            query_data = self._add_text(query_data, query_instruction, need_loss=False, enable_cfg=False, is_demo=False, segment_id=0)


        query_data = self._add_text(query_data, "Assistant:", need_loss=False, enable_cfg=False, is_demo=False, segment_id=1)


        query_data = self._add_image(
            query_data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-1]))),
            need_loss=True,
            need_vae=False,
            need_vit=False,
            enable_cfg=False,
            is_demo=False,
            segment_id=1
        )

        remaining_budget = self.max_num_tokens_per_sample - query_data['num_tokens']
        valid_demos = []
        
        # Original shot count (total samples - 1 query)
        original_shots = num_samples - 1

        # 2. Process Demos from last to first
        for sample_idx in range(num_samples - 2, -1, -1):
            base_img_idx = sample_idx * images_per_sample
            demo_data = self._init_data()
            demo_turn_id = sample_idx + 1  # 1-based demo index for CAPM


            demo_data = self._add_text(demo_data, "User:", need_loss=False, enable_cfg=False, is_demo=True, segment_id=0, demo_turn_id=demo_turn_id)


            for input_idx in range(num_inputs):
                img_idx = base_img_idx + input_idx
                demo_data = self._add_image(
                    demo_data,
                    pil_img2rgb(Image.open(io.BytesIO(row["image_list"][img_idx]))),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                    enable_cfg=False,
                    is_demo=True,
                    segment_id=0,
                    demo_turn_id=demo_turn_id
                )

            # Demo instruction -> Disable CFG
            instruction = get_instruction(row["instruction_list"][sample_idx])
            if instruction and instruction.strip():
                demo_data = self._add_text(demo_data, instruction, need_loss=False, enable_cfg=False, is_demo=True, segment_id=0, demo_turn_id=demo_turn_id)


            demo_data = self._add_text(demo_data, "Assistant:", need_loss=False, enable_cfg=False, is_demo=True, segment_id=1, demo_turn_id=demo_turn_id)


            output_img_idx = base_img_idx + num_inputs
            demo_data = self._add_image(
                demo_data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][output_img_idx]))),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
                is_demo=True,
                segment_id=1,
                demo_turn_id=demo_turn_id
            )

            if demo_data['num_tokens'] <= remaining_budget:
                valid_demos.insert(0, demo_data)
                remaining_budget -= demo_data['num_tokens']
            else:
                # Log truncation info
                kept_shots = len(valid_demos)
                print(f"[Truncate VC] {self.dataset_name}: {original_shots}-shot -> {kept_shots}-shot. Budget: {self.max_num_tokens_per_sample}")
                break

        # 3. Merge
        final_data = self._init_data()
        for demo in valid_demos:
            self._merge_data(final_data, demo)
        self._merge_data(final_data, query_data)
        return final_data
