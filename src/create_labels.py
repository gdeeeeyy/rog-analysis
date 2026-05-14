import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
IMG_DIR = "output_baseline"
YOLO_ROOT = "yolo_v26_data"
CSV_PATH = "agni-manual-logging.csv"
BBOX_CSV_PATH = "bbox_slopes.csv"

# Structure setup
for p in ["images/train", "labels/train", "images/val", "labels/val"]:
    os.makedirs(os.path.join(YOLO_ROOT, p), exist_ok=True)

def identify_previous_islands(img):
    H, W = img.shape
    blur = cv2.medianBlur(img, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    boxes = []
    for j in range(1, num_labels):
        if stats[j, cv2.CC_STAT_AREA] > (H * W * 0.0002):
            x, y, w, h = stats[j, :4]
            boxes.append([int(x), int(y), int(w), int(h)])

    return sorted(boxes, key=lambda box: (box[1], box[0]))

def identify_generate_bbox_box(img, frame_in_cycle):
    H, W = img.shape

    denoised = cv2.medianBlur(img, 5)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_size = 25 if frame_in_cycle >= 100 else 7
    kernel = np.ones((k_size, k_size), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
    min_area_threshold = (H * W) * 0.0005

    valid_kernels = []
    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]
        if area > min_area_threshold:
            x, y, w, h = stats[j, :4]
            valid_kernels.append([x, y, x + w, y + h])

    if not valid_kernels:
        return []

    valid_kernels = np.array(valid_kernels)
    x1 = np.min(valid_kernels[:, 0])
    y1 = np.min(valid_kernels[:, 1])
    x2 = np.max(valid_kernels[:, 2])
    y2 = np.max(valid_kernels[:, 3])

    pad = 15
    rx = max(0, x1 - pad)
    ry = max(0, y1 - pad)
    rw = min(W, x2 + pad) - rx
    rh = min(H, y2 + pad) - ry

    return [[int(rx), int(ry), int(rw), int(rh)]]

def boxes_to_yolo_lines(boxes, W, H):
    yolo_lines = []
    for x, y, w, h in boxes:
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        nw, nh = w / W, h / H
        yolo_lines.append(f"0 {cx} {cy} {nw} {nh}")
    return yolo_lines

def generate_hybrid_labels():
    df = pd.read_csv(CSV_PATH)
    image_ids = df['ImageNo'].tolist()

    print("Generating Hybrid YOLO labels (Previous <= 50 < generate_bbox)...")
    bbox_rows = []
    
    for i, img_no in enumerate(tqdm(image_ids)):
        filename = f"{int(img_no):05d}.png"
        img = cv2.imread(os.path.join(IMG_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        H, W = img.shape
        
        # 1. Determine Position in Cycle
        frame_in_cycle = (int(img_no) - 1) % 250 + 1
        
        # --- PHASE LOGIC ---
        if frame_in_cycle <= 50:
            # PREVIOUS MODE: Every island gets its own line.
            boxes = identify_previous_islands(img)
        else:
            # GENERATE_BBOX MODE: One box encompassing all kernels.
            boxes = identify_generate_bbox_box(img, frame_in_cycle)

        if not boxes: continue

        yolo_lines = boxes_to_yolo_lines(boxes, W, H)
        for x, y, w, h in boxes:
            bbox_rows.append([int(img_no), x, y, w, h])

        # 3. Save Data (80/20 Split)
        split = "train" if i % 5 != 0 else "val"
        cv2.imwrite(os.path.join(YOLO_ROOT, f"images/{split}", filename), img)
        with open(os.path.join(YOLO_ROOT, f"labels/{split}", f"{int(img_no):05d}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))

    pd.DataFrame(bbox_rows, columns=["imgNo", "x", "y", "w", "h"]).to_csv(BBOX_CSV_PATH, index=False)
    print(f"Saved bbox CSV to {BBOX_CSV_PATH}")

if __name__ == "__main__":
    generate_hybrid_labels()
