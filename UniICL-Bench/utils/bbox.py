"""Public release module documentation."""
import re


def parse_bbox(text):
    """Public release documentation."""
    if isinstance(text, list):
        return text

    if text is None:
        return None


    pattern = r'\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?'
    match = re.search(pattern, str(text))
    if match:
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            return coords
        except (ValueError, IndexError):
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
