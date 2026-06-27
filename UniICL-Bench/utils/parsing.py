"""Public release module documentation."""
import re


def get_instruction_text(item):
    """Public release documentation."""
    return item.get('instruction', item.get('text', ''))


def extract_answer_from_tags(text):
    """Public release documentation."""
    if not text:
        return None
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_option_letter(text):
    """Public release documentation."""
    if not text:
        return None


    patterns = [
        r'(?:answer|\\u7b54\\u6848)[\\s:\uff1a\\u662f]*([A-Z])',
        r'\b([A-Z])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def extract_action_from_tags(text):
    """Public release documentation."""
    if not text:
        return None
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_mcq_options(question_text):
    """Public release documentation."""
    options = {}

    pattern = r'([A-Z])[\.\)]\s*([^\n]+)'
    matches = re.findall(pattern, question_text)

    for letter, content in matches:
        options[letter] = content.strip()

    return options


def extract_score_from_annotation(annotation):
    """Public release documentation."""
    if isinstance(annotation, (int, float)):
        return float(annotation)

    if isinstance(annotation, str):

        try:
            return float(annotation)
        except ValueError:
            pass


        answer_match = re.search(r'<answer>\s*(\d+(?:\.\d+)?)\s*</answer>', annotation)
        if answer_match:
            return float(answer_match.group(1))



        all_numbers = re.findall(r'(\d+\.?\d*)', annotation)
        if all_numbers:
            return float(all_numbers[-1])

    return None


def extract_label_from_annotation(annotation):
    """Public release documentation."""
    if isinstance(annotation, str):

        answer_match = re.search(r'<answer>\s*([^<]+?)\s*</answer>', annotation, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()


        return annotation.strip()
    return str(annotation)


def parse_option_label(text, valid_labels, strict_action=False):
    """Public release documentation."""
    if not text:
        return None

    text_lower = text.lower()


    if strict_action:
        action = extract_action_from_tags(text)
        if action and action in valid_labels:
            return action



    sorted_labels = sorted(valid_labels, key=len, reverse=True)

    for label in sorted_labels:
        if label.lower() in text_lower:
            return label

    return None


def _extract_label_from_text(text, options):
    """Public release documentation."""
    if not text:
        return None

    text_lower = text.lower().strip()


    sorted_options = sorted(options, key=lambda x: len(str(x)), reverse=True)

    for option in sorted_options:
        option_str = str(option).lower().strip()
        if option_str in text_lower:
            return option


    for option in sorted_options:
        option_str = str(option).lower().strip()

        pattern = r'\b' + re.escape(option_str) + r'\b'
        if re.search(pattern, text_lower):
            return option

    return None
