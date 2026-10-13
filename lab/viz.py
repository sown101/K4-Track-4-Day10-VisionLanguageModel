"""Vẽ box dự đoán (nét liền) và box đúng (nét đứt mảnh) lên ảnh — dùng cho ảnh minh hoạ trong REPORT."""
from PIL import Image, ImageDraw

from lab.labels import yolo_to_pixel

COLORS = {"pothole": "red", "crack": "yellow", "manhole": "cyan"}


def _dashed_rect(d, box, color, dash=4):
    x1, y1, x2, y2 = box
    for (a, b, c, e) in [(x1, y1, x2, y1), (x2, y1, x2, y2), (x1, y2, x2, y2), (x1, y1, x1, y2)]:
        n = max(int(max(abs(c - a), abs(e - b)) // dash), 1)
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1)
            d.line([(a + (c - a) * t0, b + (e - b) * t0), (a + (c - a) * t1, b + (e - b) * t1)],
                   fill=color, width=1)


def draw_pred_gt(image_path, rec, pred, coord="norm1000"):
    """pred: list[{"label","bbox"}] từ scripts.score.parse_output (hoặc None nếu JSON hỏng)."""
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = rec["width"], rec["height"]
    for o in rec["objects"]:
        _dashed_rect(d, yolo_to_pixel(o["yolo"], W, H), COLORS.get(o["label"], "white"))
    sx, sy = (W / 1000, H / 1000) if coord == "norm1000" else (1, 1)
    for p in pred or []:
        x1, y1, x2, y2 = p["bbox"]
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        color = COLORS.get(p["label"], "magenta")   # magenta = nhãn ngoài 3 lớp
        d.rectangle(box, outline=color, width=2)
        d.text((box[0] + 2, max(box[1] - 11, 0)), p["label"], fill=color)
    return img
