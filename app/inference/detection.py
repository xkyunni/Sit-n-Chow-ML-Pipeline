from ultralytics import YOLO

DOG_CLASS_ID_COCO = 16

# restricts value(v) between 'lo' and 'hi' bounds
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# crop with padding to ensure all features are within the new img
def crop_with_padding(frame, xyxy, pad_frac=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)

    bw, bh = (x2 - x1), (y2 - y1)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac

    cx1 = int(clamp(x1 - pad_x, 0, w - 1))
    cy1 = int(clamp(y1 - pad_y, 0, h - 1))
    cx2 = int(clamp(x2 + pad_x, 0, w - 1))
    cy2 = int(clamp(y2 + pad_y, 0, h - 1))

    if cx2 <= cx1 or cy2 <= cy1:
        return None, (cx1, cy1, cx2, cy2)

    return frame[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)

class DogDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.10, pad_frac: float = 0.15):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.pad_frac = pad_frac

    # run YOLO detection and return dog detection with highest confidence for 1 frame
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return None

        best_xyxy = None
        best_conf = -1.0

        for b in boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            if cls == DOG_CLASS_ID_COCO and conf >= self.conf_threshold and conf > best_conf:
                best_conf = conf
                best_xyxy = b.xyxy[0].tolist()

        if best_xyxy is None:
            return None

        # crop img for classifier
        crop, padded_box = crop_with_padding(frame, best_xyxy, self.pad_frac)
        if crop is None:
            return None

        return {
            "bbox": [int(v) for v in padded_box], # bbox overlay
            "det_confidence": best_conf,
            "crop": crop,
        }