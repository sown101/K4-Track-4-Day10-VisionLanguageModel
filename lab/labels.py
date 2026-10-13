"""Bước 2: nhãn YOLO -> JSON grounding (thang 0–1000 hoặc pixel) và prompt.

Tách ra file riêng để notebook E0–E5 cùng import một nguồn duy nhất
(nhãn lúc train và prompt lúc suy luận phải khớp nhau tuyệt đối).
"""
import json

CLASSES = ("pothole", "crack", "manhole")

PROMPTS = {
    "norm1000": (
        "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole.\n"
        "Return ONLY a JSON list, no extra text. Each item: "
        '{"label": <class>, "bbox_2d": [x1, y1, x2, y2]}.\n'
        "Coordinates are integers on a 0-1000 scale relative to the image width and height "
        "(0 = left/top edge, 1000 = right/bottom edge).\n"
        "If there is no damage, return []."
    ),
    "pixel": (
        "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole.\n"
        "Return ONLY a JSON list, no extra text. Each item: "
        '{"label": <class>, "bbox_2d": [x1, y1, x2, y2]}.\n'
        "Coordinates are integer pixel positions in the {w}x{h} image "
        "(x from 0 to {w}, y from 0 to {h}).\n"
        "If there is no damage, return []."
    ),
}


def build_prompt(coord="norm1000", width=640, height=360):
    p = PROMPTS[coord]
    return p.replace("{w}", str(width)).replace("{h}", str(height)) if coord == "pixel" else p


def _scale(coord, width, height):
    if coord == "norm1000":
        return 1000, 1000
    if coord == "pixel":
        return width, height
    raise ValueError(f"coord không hợp lệ: {coord}")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) -> {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = _scale(coord, width, height)
    x1 = min(max(round((xc - w / 2) * sx), 0), sx)
    y1 = min(max(round((yc - h / 2) * sy), 0), sy)
    x2 = min(max(round((xc + w / 2) * sx), 0), sx)
    y2 = min(max(round((yc + h / 2) * sy), 0), sy)
    # Box cực nhỏ có thể làm tròn thành x1 == x2; score.parse_output coi cả ảnh là hỏng
    # nếu x2 <= x1, nên nới tối thiểu 1 đơn vị (ưu tiên nới về phía còn chỗ).
    if x2 <= x1:
        x1, x2 = (x1, x1 + 1) if x1 < sx else (x1 - 1, x1)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < sy else (y1 - 1, y1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    # trên -> dưới, trái -> phải: thứ tự ổn định để model học
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return boxes


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    """Chuỗi JSON đáp án gọn (không khoảng trắng); "[]" nếu ảnh không có object."""
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def box_to_pixel(bbox, coord="norm1000", width=640, height=360):
    """Quy bbox_2d về pixel (float) để vẽ / kiểm tra round-trip."""
    sx, sy = _scale(coord, width, height)
    x1, y1, x2, y2 = bbox
    return (x1 / sx * width, y1 / sy * height, x2 / sx * width, y2 / sy * height)


def yolo_to_pixel(yolo, width=640, height=360):
    xc, yc, w, h = yolo
    return ((xc - w / 2) * width, (yc - h / 2) * height,
            (xc + w / 2) * width, (yc + h / 2) * height)
