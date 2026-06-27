"""Public release module documentation."""

from .scoring import (
    load_qalign_model,
    compute_qalign_score,
    compute_qalign_batch_scores,
    load_hpsv3_model,
    compute_hpsv3_score,
)

from .bbox import (
    parse_bbox,
    normalize_bbox,
    compute_iou,
)

from .parsing import (
    parse_option_label,
    extract_answer_from_tags,
    extract_option_letter,
    extract_action_from_tags,
    parse_mcq_options,
    extract_score_from_annotation,
    extract_label_from_annotation,
    get_instruction_text,
)

from .judge import (
    call_vllm_judge,
    mllm_assisted_extraction,
)

from .icl import (
    build_icl_input,
)

from .evaluators import (
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

__all__ = [
    # Scoring
    'load_qalign_model',
    'compute_qalign_score',
    'compute_qalign_batch_scores',
    'load_hpsv3_model',
    'compute_hpsv3_score',
    # BBox
    'parse_bbox',
    'normalize_bbox',
    'compute_iou',
    # Parsing
    'parse_option_label',
    'extract_answer_from_tags',
    'extract_option_letter',
    'extract_action_from_tags',
    'parse_mcq_options',
    'extract_score_from_annotation',
    'extract_label_from_annotation',
    'get_instruction_text',
    # Judge
    'call_vllm_judge',
    'mllm_assisted_extraction',
    # ICL
    'build_icl_input',
    # Evaluators
    'eval_grounding',
    'eval_attr_rec_gen',
    'eval_vqa_gen',
    'eval_caption_styled',
    'eval_t2i',
    'eval_i2i_editing',
    'eval_aesthetic_assessment',
    'eval_authenticity_detection',
    'eval_image_perfection',
    'eval_fcb_classification',
    'eval_fci_t2i',
    'eval_planning',
    'eval_visualcloze_g',
    'eval_visualcloze_u',
    'eval_chain_of_editing',
]
