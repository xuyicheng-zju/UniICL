# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Public release module documentation."""

import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .vlm_dataset import SftJSONLIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class SftJSONLIterableDatasetWithGen(SftJSONLIterableDataset):
    """Public release documentation."""

    def __init__(
        self, dataset_name, transform, tokenizer, frame_sampler,
        jsonl_path_list, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False, shuffle_seed=0,
        max_num_tokens_per_sample=None, # Accept but ignore
        **kwargs
    ):
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        super().__init__(
            dataset_name, transform, tokenizer, frame_sampler,
            jsonl_path_list, data_dir_list, num_used_data,
            local_rank, world_size, num_workers, data_status,
            shuffle_lines, shuffle_seed,
            **kwargs
        )

    def change_format(self, data, num_images):
        """Public release documentation."""
        elements = []
        image_idx = 0
        for conv_idx, conversation in enumerate(data['conversations']):
            if conversation['from'] == 'human':



                value = conversation['value']



                user_turns = value.count('\nUser:') + (1 if value.startswith('User:') else 0)
                obs_turns = value.count('\nObservation:') + (1 if value.startswith('Observation:') else 0)


                if user_turns > 0 and obs_turns > 0:
                    return None  # Mixed format
                elif user_turns == 0 and obs_turns == 0:
                    return None  # No valid format


                if obs_turns > 0:

                    main_marker = 'Observation:'
                    parts = value.split('Observation:')
                else:

                    main_marker = 'User:'
                    parts = value.split('User:')
                parts = [p.strip() for p in parts if p.strip()]



                has_demos = (user_turns > 1 or obs_turns > 1)
                for part_idx, part in enumerate(parts):

                    is_demo = 1 if (has_demos and part_idx < len(parts) - 1) else 0


                    if 'Assistant:' in part:
                        user_part, assistant_part = part.split('Assistant:', 1)


                        user_part_with_marker = main_marker + ' ' + user_part.strip()

                        if '<image>' not in user_part_with_marker:
                            if user_part_with_marker.strip():
                                elements.append({
                                    'type': 'text',
                                    'has_loss': 0,
                                    'text': user_part_with_marker.strip(),
                                    'is_demo': is_demo,
                                    'segment_id': 0,  # User
                                    'demo_turn_id': part_idx if is_demo else -1,
                                })
                        else:
                            text_list = user_part_with_marker.split('<image>')
                            for idx, text in enumerate(text_list):
                                if text.strip() != '':
                                    elements.append({
                                        'type': 'text',
                                        'has_loss': 0,
                                        'text': text.strip(),
                                        'is_demo': is_demo,
                                        'segment_id': 0,
                                        'demo_turn_id': part_idx if is_demo else -1,
                                    })
                                if (idx != len(text_list) - 1) and (image_idx < num_images):
                                    elements.append({
                                        'type': 'vit_image',
                                        'image_idx': image_idx,
                                        'is_demo': is_demo,
                                        'segment_id': 0,
                                        'demo_turn_id': part_idx if is_demo else -1,
                                    })
                                    image_idx += 1


                        if is_demo and assistant_part.strip():
                            elements.append({
                                'type': 'text',
                                'has_loss': 0,
                                'text': 'Assistant: ' + assistant_part.strip(),
                                'is_demo': is_demo,
                                'segment_id': 1,  # Assistant
                                'demo_turn_id': part_idx if is_demo else -1,
                            })
                    else:

                        part_with_marker = main_marker + ' ' + part.strip()

                        if '<image>' not in part_with_marker:
                            if part_with_marker.strip():
                                elements.append({
                                    'type': 'text',
                                    'has_loss': 0,
                                    'text': part_with_marker.strip(),
                                    'is_demo': is_demo,
                                    'segment_id': 0,
                                    'demo_turn_id': part_idx if is_demo else -1,
                                })
                        else:
                            text_list = part_with_marker.split('<image>')
                            for idx, text in enumerate(text_list):
                                if text.strip() != '':
                                    elements.append({
                                        'type': 'text',
                                        'has_loss': 0,
                                        'text': text.strip(),
                                        'is_demo': is_demo,
                                        'segment_id': 0,
                                        'demo_turn_id': part_idx if is_demo else -1,
                                    })
                                if (idx != len(text_list) - 1) and (image_idx < num_images):
                                    elements.append({
                                        'type': 'vit_image',
                                        'image_idx': image_idx,
                                        'is_demo': is_demo,
                                        'segment_id': 0,
                                        'demo_turn_id': part_idx if is_demo else -1,
                                    })
                                    image_idx += 1

            elif conversation['from'] == 'gpt':

                is_demo = 0
                segment_id = 1  # Assistant


                elements.append({
                    'type': 'text',
                    'has_loss': 1,
                    'text': conversation['value'],
                    'is_demo': is_demo,
                    'segment_id': segment_id,
                    'demo_turn_id': -1,
                })

        return elements

    def __iter__(self):
        """Public release documentation."""
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    data_item = json.loads(data)
                    raw_images = None
                    if 'image' in data_item:
                        if type(data_item['image']) == list:
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, image)))
                                for image in data_item['image']
                            ]
                        else:
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, data_item['image'])))
                            ]
                    elif 'video' in data_item:
                        raw_images = self.frame_sampler(os.path.join(image_dir, data_item['video']))
                        special_tokens = '<image>' * len(raw_images)
                        for item in data_item['conversations']:
                            if '<video>' in item['value']:
                                item['value'] = item['value'].replace('<video>', special_tokens)
                                break
                            else:
                                raise ValueError("Cannot find <video> in the conversation!")
                except:
                    traceback.print_exc()
                    continue


                elements = self.change_format(data_item, len(raw_images) if raw_images else 0)


                if elements is None:
                    continue


                element_infos = []
                demo_stats = {}
                total_tokens = 0

                for item in elements:
                    if item['type'] == 'text':
                        text_data = item['text']
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) == 0:
                            continue
                        token_len = len(text_ids)
                        info = {
                            'type': 'text',
                            'text_ids': text_ids,
                            'token_len': token_len,
                            'has_loss': item['has_loss'],
                            'is_demo': item.get('is_demo', 0),
                            'segment_id': item.get('segment_id', 0),
                            'demo_turn_id': item.get('demo_turn_id', -1),
                        }
                        element_infos.append(info)
                        total_tokens += token_len
                        if info['is_demo'] == 1:
                            stats = demo_stats.setdefault(info['demo_turn_id'], {'tokens': 0, 'count': 0})
                            stats['tokens'] += token_len
                            stats['count'] += 1

                    elif item['type'] == 'vit_image':

                        img_idx = item['image_idx']
                        if raw_images and img_idx < len(raw_images):
                            image_tensor = self.transform(raw_images[img_idx], img_num=len(raw_images))
                            height, width = image_tensor.shape[1:]
                            num_img_tokens = width * height // transform_stride ** 2
                            info = {
                                'type': 'vit_image',
                                'image_tensor': image_tensor,
                                'token_len': num_img_tokens,
                                'has_loss': 0,
                                'is_demo': item.get('is_demo', 0),
                                'segment_id': item.get('segment_id', 0),
                                'demo_turn_id': item.get('demo_turn_id', -1),
                            }
                            element_infos.append(info)
                            total_tokens += num_img_tokens
                            if info['is_demo'] == 1:
                                stats = demo_stats.setdefault(info['demo_turn_id'], {'tokens': 0, 'count': 0})
                                stats['tokens'] += num_img_tokens
                                stats['count'] += 1

                total_len = total_tokens + 2 * len(element_infos)
                budget = self.max_num_tokens_per_sample

                if budget is not None and budget > 0 and total_len > budget:
                    if len(demo_stats) == 0:
                        continue  # No demos to truncate, skip
                    curr_len = total_len
                    drop_ids = set()

                    for demo_id in sorted(demo_stats.keys()):
                        stats = demo_stats[demo_id]
                        curr_len -= stats['tokens'] + 2 * stats['count']
                        drop_ids.add(demo_id)
                        if curr_len <= budget:
                            break
                    if curr_len > budget:
                        continue  # Still over budget after truncation
                    if drop_ids:
                        element_infos = [
                            info for info in element_infos
                            if info.get('demo_turn_id', -1) not in drop_ids
                        ]


                for info in element_infos:
                    if info['type'] == 'text':
                        text_ids_list.append(info['text_ids'])
                        num_tokens += info['token_len']
                        current_plan = {
                            'type': 'text',
                            'enable_cfg': 0,
                            'loss': info['has_loss'],
                            'special_token_loss': 0,
                            'special_token_label': None,
                            'is_demo': info['is_demo'],
                            'segment_id': info['segment_id'],
                            'demo_turn_id': info.get('demo_turn_id', 0),
                        }
                        sequence_plan.append(current_plan)
                    elif info['type'] == 'vit_image':
                        image_tensor_list.append(info['image_tensor'])
                        num_tokens += info['token_len']
                        current_plan = {
                            'type': 'vit_image',
                            'enable_cfg': 0,
                            'loss': 0,
                            'special_token_loss': 0,
                            'special_token_label': None,
                            'is_demo': info['is_demo'],
                            'segment_id': info['segment_id'],
                            'demo_turn_id': info.get('demo_turn_id', 0),
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    continue  # No loss tokens

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
