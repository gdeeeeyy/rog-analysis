# pyrefly: ignore [missing-import]
import cv2
import numpy as np

def get_yolo_bounding_boxes(img_gray, frame_in_cycle, pre_merge=True):
    """
    Generates bounding boxes from original grayscale images.
    Aligns with multi-point visualization: returns individual boxes for each component.
    """
    if img_gray is None: return []
    H, W = img_gray.shape

    # 1. DENOISE
    denoised = cv2.medianBlur(img_gray, 5)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if pre_merge:
        # Pre-merge Mode: Sensitive island detection (Smaller threshold, No morphology)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        min_area = (H * W * 0.0002)
        bboxes = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                x, y, w, h = stats[i, :4]
                bboxes.append((int(x), int(y), int(w), int(h)))
        return sorted(bboxes, key=lambda b: (b[1], b[0]))
    else:
        # Merged Mode: Closing logic (Larger threshold + Morphology)
        k_size = 25 if frame_in_cycle >= 100 else 7
        kernel = np.ones((k_size, k_size), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
        min_area = (H * W * 0.0005)
        
        bboxes = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                x, y, w, h = stats[i, :4]
                # No padding to ensure boxes contain ONLY the flame
                bboxes.append((int(x), int(y), int(w), int(h)))
        
        return sorted(bboxes, key=lambda b: (b[1], b[0]))

def get_mask_metrics(mask_img):
    """Calculates precise area and perimeter from a binary or outline mask."""
    if mask_img is None: return 0, 0
    
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    
    area = int(np.sum(filled) // 255)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) if contours else 0
    
    return area, perimeter
