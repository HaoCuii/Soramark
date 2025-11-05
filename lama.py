from pathlib import Path
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
subprocess.run(["iopaint", "--help"], capture_output=True, check=True)

MODEL_PATH = Path("best.pt")
IMAGE_PATH = Path("in.jpg")
OUTPUT_PATH = Path("out.png")
MASK_PATH = Path("mask.png")  



model = YOLO(str(MODEL_PATH))
result = model.predict(source=str(IMAGE_PATH), imgsz=896, conf=0.35, verbose=False)[0]

if result.boxes is None or len(result.boxes) == 0:
    sys.exit("No watermark detected.")

#Pick most confident box 
img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]
i = int(result.boxes.conf.argmax())
x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)

#Bounds
x1, y1 = max(0, x1), max(0, y1)
x2, y2 = min(w - 1, x2), min(h - 1, y2)
print(f"[INFO] Watermark bbox: ({x1}, {y1}) → ({x2}, {y2})")

mask = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
cv2.imwrite(str(MASK_PATH), mask)

use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
device = "cuda" if use_cuda else "cpu"

cmd = [
    "iopaint", "run",
    "--model", "lama",
    "--device", device,
    "--image", str(IMAGE_PATH),
    "--mask", str(MASK_PATH),
    "--output", str(OUTPUT_PATH)
]

result = subprocess.run(cmd, capture_output=True, text=True)

