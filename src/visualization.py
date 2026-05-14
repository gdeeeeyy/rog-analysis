import cv2
import numpy as np
import imageio.v2 as imageio

def draw_dotted_rect(img, pt1, pt2, color, thickness=2, gap=12):
    """Draws a bold dotted rectangle (Standard Pipeline Style)"""
    x1, y1 = pt1
    x2, y2 = pt2
    pts = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for start, end in pts:
        dist = np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
        if dist == 0: continue
        ux, uy = (end[0]-start[0])/dist, (end[1]-start[1])/dist
        curr_dist = 0
        while curr_dist < dist:
            sx, sy = int(start[0]+curr_dist*ux), int(start[1]+curr_dist*uy)
            ex, ey = int(start[0]+min(curr_dist+gap/2, dist)*ux), int(start[1]+min(curr_dist+gap/2, dist)*uy)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
            curr_dist += gap

def overlay_mask_boundary(img, mask_img):
    """Overlays the red flame boundary on the image."""
    if mask_img is None: return img
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2) # Red
    return img

def draw_pre_merge_frame(img, bboxes, mask_img, img_no, perimeter=0):
    """Draws dotted cyan boxes and red boundary for multiple flame points."""
    vis_img = img.copy()
    if mask_img is not None:
        vis_img = overlay_mask_boundary(vis_img, mask_img)
        
    for b in bboxes:
        draw_dotted_rect(vis_img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 255), 2)
    
    # NEAT TOP-LEFT TEXT (No 'Pre-merge' or 'Merged' labels)
    text = f"Frame: {img_no} | P: {int(perimeter)} | Pts: {len(bboxes)}"
    cv2.putText(vis_img, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis_img

def draw_post_merge_frame(img, bboxes, mask_img, img_no, perimeter=0):
    """Draws green bounding boxes and red boundary for the flame."""
    vis_img = img.copy()
    if mask_img is not None:
        vis_img = overlay_mask_boundary(vis_img, mask_img)
        
    for b in bboxes:
        x, y, w, h = b
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # NEAT TOP-LEFT TEXT (No 'Merged' label)
    text = f"Frame: {img_no} | P: {int(perimeter)}"
    cv2.putText(vis_img, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis_img

def save_cycle_gif(frames, output_path):
    """Compiles the annotated frames into a GIF."""
    if frames:
        imageio.mimsave(output_path, frames, duration=0.1)
