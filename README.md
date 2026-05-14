# Flame Propagation Analysis Pipeline (Stage 4)

This pipeline performs modular flame propagation analysis using a hybrid YOLO-style bounding box detection and precise mask-based metrics.

## Project Structure

```text
stage4-impl/
├── run_rog.py          # Main execution script
├── src/                # Project modules
│   ├── __init__.py
│   ├── analytics.py    # RoG and Frame Analytics
│   ├── image_processing.py # YOLO and Mask processing
│   ├── visualization.py # GIF and Overlay generation
│   ├── utils.py        # CSV and File utilities
│   └── ... (other modules)
└── venv/               # Virtual environment
```

## Setup

1. **Virtual Environment**: The project uses a dedicated `venv`.
   ```bash
   source venv/bin/activate
   ```

2. **Dependencies**: Ensure all requirements are installed (opencv, pandas, matplotlib, imageio, tqdm).

## How to Run

Execute the main runner from the root directory:

```bash
python run_rog.py \
    --image_dir /path/to/original/images \
    --mask_dir /path/to/flame/masks \
    --output_dir /path/to/Analysis_Results \
    --csv_file /path/to/agni-manual-logging.csv \
    --frame_analytics
```

### Parameters:
- `--image_dir`: (Required) Path to the folder containing original grayscale frames.
- `--mask_dir`: (Optional) Path to the folder containing flame boundary masks (outlines). If provided, the pipeline calculates precise Area and Perimeter.
- `--output_dir`: (Required) Path where results (plots, GIFs, CSVs) will be saved.
- `--csv_file`: (Optional) Path to the manual logging CSV to filter frames by tag (e.g., 'Usable').
- `--process_tags`: (Optional) Comma-separated tags to process (default: `Usable`).
- `--frame_analytics`: (Flag) Generate additional scatter plots for first/last merged frames.

## Outputs

- **`combined_rog_results.csv`**: Frame-by-frame metrics including Area, Perimeter, and Growth Rates.
- **`bbox_slopes.csv`**: Linear regression slopes for flame growth per cycle.
- **`vis/`**: GIFs showing the original frames with overlaid bounding boxes (dotted cyan/yellow for pre-merge, solid green for merged) and red flame boundaries.
- **`plots/`**: Trend plots for Perimeter, Area, Length, Breadth, and Growth Rates.
- **`frame_analytics/`**: Detailed scatter plots and averages for first and last merged frames.
