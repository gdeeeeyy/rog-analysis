import cv2
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

def draw_dotted_rect(img, pt1, pt2, color, thickness=1, gap=10):
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

def estimate_boundaries(args):
    print(f"Loading data from {args.bbox_csv}...")
    df = pd.read_csv(args.bbox_csv)
    
    # 1. CALCULATE SLOPES PER CYCLE
    # Assume 250 frames per cycle, reference at frame 78
    df['Cycle'] = ((df['imgNo'] - 1) // 250) + 1
    df['FrameInCycle'] = ((df['imgNo'] - 1) % 250) + 1
    
    cycle_slopes = []
    
    print("Calculating growth slopes per cycle...")
    for cycle, group in df.groupby('Cycle'):
        # Only use usable frames for slope calculation (e.g. FrameInCycle >= 50 and <= 200)
        fit_data = group[(group['FrameInCycle'] >= 60) & (group['FrameInCycle'] <= 180)]
        if len(fit_data) < 10: continue
        
        # Linear regression for Length (w) and Breadth (h)
        slope_l, intercept_l = np.polyfit(fit_data['FrameInCycle'], fit_data['w'], 1)
        slope_b, intercept_b = np.polyfit(fit_data['FrameInCycle'], fit_data['h'], 1)
        
        # Get Reference BBox at frame 78 (or closest)
        ref_row = group[group['FrameInCycle'] >= 78].iloc[0] if not group[group['FrameInCycle'] >= 78].empty else group.iloc[len(group)//2]
        
        cycle_slopes.append({
            'Cycle': cycle,
            'Slope_L': slope_l,
            'Slope_B': slope_b,
            'Ref_X': ref_row['x'],
            'Ref_Y': ref_row['y'],
            'Ref_W': ref_row['w'],
            'Ref_H': ref_row['h'],
            'Ref_Frame': ref_row['FrameInCycle']
        })
    
    slopes_df = pd.DataFrame(cycle_slopes)
    slopes_df.to_csv(os.path.join(args.output_dir, "calculated_slopes.csv"), index=False)
    print(f"Saved slopes to {args.output_dir}/calculated_slopes.csv")

    # 2. GENERATE ESTIMATED BOUNDARIES
    os.makedirs(os.path.join(args.output_dir, "boundary_estimates"), exist_ok=True)
    
    results = []
    print("Generating boundary estimates...")
    
    for _, row in slopes_df.iterrows():
        cycle = int(row['Cycle'])
        cx, cy = row['Ref_X'] + row['Ref_W']/2, row['Ref_Y'] + row['Ref_H']/2
        
        start_frame = (cycle - 1) * 250 + 1
        end_frame = cycle * 250
        
        limit_frames = args.limit if args.limit else 250
        
        for f_no in tqdm(range(start_frame, start_frame + limit_frames)):
            filename = f"{str(f_no).zfill(5)}.png"
            img_path = os.path.join(args.img_dir, filename)
            if not os.path.exists(img_path): continue
            
            dt = (f_no - (start_frame - 1)) - row['Ref_Frame']
            
            # ESTIMATION LOGIC (with fix for negative dimensions)
            est_l = max(10, row['Ref_W'] + dt * row['Slope_L'])
            est_b = max(10, row['Ref_H'] + dt * row['Slope_B'])
            est_x, est_y = int(cx - est_l / 2), int(cy - est_b / 2)
            
            results.append({
                "ImageNo": f_no,
                "Cycle": cycle,
                "Est_X": est_x, "Est_Y": est_y, "Est_W": int(est_l), "Est_H": int(est_b)
            })
            
            if args.visualize:
                img = cv2.imread(img_path)
                if img is not None:
                    draw_dotted_rect(img, (est_x, est_y), (est_x + int(est_l), est_y + int(est_b)), (0, 255, 255), 2)
                    cv2.imwrite(os.path.join(args.output_dir, "boundary_estimates", filename), img)

    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "estimated_boundaries.csv"), index=False)
    print(f"Done. Estimated boundaries saved to {args.output_dir}/estimated_boundaries.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate flame boundaries using growth slopes")
    parser.add_argument("--img_dir", default="output_baseline", help="Source images")
    parser.add_argument("--bbox_csv", default="bbox_slopes.csv", help="Input bounding boxes")
    parser.add_argument("--output_dir", default="SlopeEstimation_Results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visual overlays")
    parser.add_argument("--limit", type=int, help="Limit frames per cycle for visualization")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    estimate_boundaries(args)
