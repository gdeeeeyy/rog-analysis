import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm

from src import utils
from src import image_processing
from src import visualization
from src import analytics

def main():
    parser = argparse.ArgumentParser(description="Unified Modular Flame Analysis (Perimeter Focused)")
    parser.add_argument('--image_dir', required=True, help="Original images dir")
    parser.add_argument('--mask_dir', default='', help="Masks (outlines) dir (optional, for precise metrics)")
    parser.add_argument('--output_dir', required=True, help="Output dir")
    parser.add_argument('--csv_file', default='', help="Agni manual logging CSV (optional)")
    parser.add_argument('--process_tags', default='Usable', help="Comma-separated tags to process")
    parser.add_argument('--frame_analytics', action='store_true', help="Generate scatter plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Group images into cycles
    cycle_files = utils.get_cycle_files(args.image_dir, args.csv_file, args.process_tags)
    
    if not cycle_files:
        print("No valid files found to process.")
        return

    all_data = []

    # 2. Process each cycle
    for cycle, cfiles in tqdm(sorted(cycle_files.items()), desc="Processing Cycles"):
        cfiles.sort(key=lambda x: x[0])
        
        cycle_data = []
        gif_frames = []
        has_merged = False
        
        for num, fname in cfiles:
            rel_frame = (num - 1) % 250 + 1
            img_path = os.path.join(args.image_dir, fname)
            mask_path = os.path.join(args.mask_dir, fname) if args.mask_dir else None
            
            if not os.path.exists(img_path): continue
            
            orig_img = cv2.imread(img_path)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path and os.path.exists(mask_path) else None
            
            # Extract features
            if not has_merged:
                bboxes = image_processing.get_yolo_bounding_boxes(img_gray, rel_frame, pre_merge=True)
                if len(bboxes) == 1:
                    merged_bboxes = image_processing.get_yolo_bounding_boxes(img_gray, rel_frame, pre_merge=False)
                    if len(merged_bboxes) == 1:
                        has_merged = True
                        bboxes = merged_bboxes
            else:
                bboxes = image_processing.get_yolo_bounding_boxes(img_gray, rel_frame, pre_merge=False)
            
            if not bboxes: continue
            
            mask_area, mask_perim = image_processing.get_mask_metrics(mask_img)
            
            # Metrics
            min_x = min(b[0] for b in bboxes)
            min_y = min(b[1] for b in bboxes)
            max_x = max(b[0]+b[2] for b in bboxes)
            max_y = max(b[1]+b[3] for b in bboxes)
            w_union, h_union = max_x - min_x, max_y - min_y
            final_area = mask_area if mask_area > 0 else (w_union * h_union)
            final_perim = mask_perim if mask_perim > 0 else 2 * (w_union + h_union)
            
            if not has_merged:
                vis_img = visualization.draw_pre_merge_frame(orig_img, bboxes, mask_img, num, final_perim)
                is_merged_flag = False
            else:
                vis_img = visualization.draw_post_merge_frame(orig_img, bboxes, mask_img, num, final_perim)
                is_merged_flag = True
                
            entry = {
                'Cycle': cycle, 'ImageNo': num, 'RelativeFrame': rel_frame,
                'BBox_Length': w_union, 'BBox_Breadth': h_union, 'BBox_Area': w_union * h_union,
                'Precise_Area': final_area, 'Precise_Perim': final_perim,
                'IsMerged': is_merged_flag
            }
            
            cycle_data.append(entry)
            gif_frames.append(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            
        c_df = pd.DataFrame(cycle_data)
        if not c_df.empty:
            all_data.extend(c_df.to_dict('records'))
            
        visualization.save_cycle_gif(gif_frames, os.path.join(vis_dir, f"cycle_{cycle}.gif"))

    # 3. Analytics and Reporting
    final_df = pd.DataFrame(all_data)
    if not final_df.empty:
        final_df = analytics.calculate_growth_rates(final_df)
        final_df.to_csv(os.path.join(args.output_dir, "combined_rog_results.csv"), index=False)
        analytics.calculate_slopes(final_df, args.output_dir)
        
        plot_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Trend Plots
        analytics.plot_rog_data(final_df, os.path.join(plot_dir, "combined_bbox_length_plot.png"), "BBox Length Trend", 'BBox_Length')
        analytics.plot_rog_data(final_df, os.path.join(plot_dir, "combined_bbox_breadth_plot.png"), "BBox Breadth Trend", 'BBox_Breadth')
        analytics.plot_rog_data(final_df, os.path.join(plot_dir, "combined_perimeter_plot.png"), "Flame Perimeter Trend", 'Precise_Perim')
        analytics.plot_rog_data(final_df, os.path.join(plot_dir, "combined_bbox_area_plot.png"), "Flame Area Trend", 'Precise_Area')
        
        # RoG Plots
        analytics.plot_rog_data(final_df, os.path.join(plot_dir, "combined_growth_rate_perim_plot.png"), "Perimeter Growth Rate (%)", 'GrowthRate_Perim')
        analytics.plot_rog_data(final_df, os.path.join(plot_dir, "combined_growth_rate_area_plot.png"), "Area Growth Rate (%)", 'GrowthRate_Area')
        
        if args.frame_analytics:
            analytics.plot_frame_analytics(final_df, os.path.join(args.output_dir, "frame_analytics"))
            
    print(f"Pipeline complete! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
