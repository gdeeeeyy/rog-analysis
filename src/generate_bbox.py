import cv2
import numpy as np
import os
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

def identify_multi_kernel_bbox(img_path, img_no):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    H_img, W_img = img.shape
    frame_in_cycle = (int(img_no) - 1) % 250 + 1

    # 1. DENOISE
    denoised = cv2.medianBlur(img, 5)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. BRIDGE GAPS (Use larger kernel for later frames to ensure rings stay connected)
    k_size = 25 if frame_in_cycle >= 100 else 7
    kernel = np.ones((k_size, k_size), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 3. IDENTIFY ALL POTENTIAL KERNELS
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    
    # Threshold: An object must be at least 0.05% of the image area to be a 'flame'
    # This ignores the small white noise dots/sparks
    min_area_threshold = (H_img * W_img) * 0.0005 
    
    valid_kernels = []
    for i in range(1, num_labels): # Skip background (index 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area_threshold:
            x, y, w, h = stats[i, :4]
            valid_kernels.append([x, y, x + w, y + h])

    if not valid_kernels: return None

    # 4. COMPUTE ENCOMPASSING BOX (Min/Max of all kernels)
    valid_kernels = np.array(valid_kernels)
    x1 = np.min(valid_kernels[:, 0])
    y1 = np.min(valid_kernels[:, 1])
    x2 = np.max(valid_kernels[:, 2])
    y2 = np.max(valid_kernels[:, 3])

    # 5. PADDING
    pad = 15
    rx, ry = max(0, x1 - pad), max(0, y1 - pad)
    rw, rh = min(W_img, x2 + pad) - rx, min(H_img, y2 + pad) - ry
    
    return [int(rx), int(ry), int(rw), int(rh)]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", default="/scratch/ananyaa/AGNI/output_baseline")
    p.add_argument("--usable_csv", default="/scratch/ananyaa/AGNI/final_usable_unusable_images.csv")
    p.add_argument("--output_bbox_csv", default="/scratch/ananyaa/AGNI/bbox/bbox_coordinates.csv")
    args = p.parse_args()

    viz_dir = Path(args.output_bbox_csv).parent / "bboxes_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.usable_csv)
    # Process ALL images (removed the 'usable' filter as per your request)
    print(f"Generating Multi-Kernel BBoxes...")
    
    results = []
    for img_no in tqdm(df['ImageNo']):
        filename = f"{int(img_no):05d}.png"
        path = os.path.join(args.img_dir, filename)
        bbox = identify_multi_kernel_bbox(path, img_no)
        
        if bbox:
            results.append([int(img_no), bbox[0], bbox[1], bbox[2], bbox[3]])
            orig_img = cv2.imread(path)
            if orig_img is not None:
                cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 2)
                cv2.imwrite(str(viz_dir / filename), orig_img)

    pd.DataFrame(results, columns=['ImageNo', 'x', 'y', 'w', 'h']).to_csv(args.output_bbox_csv, index=False)