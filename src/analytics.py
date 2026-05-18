import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_growth_rates(df):
    """Calculates frame-to-frame percentage growth rates for both Area and Perimeter."""
    if df.empty: return df
    
    df = df.sort_values(['Cycle', 'RelativeFrame'])
    df['GrowthRate_Area'] = 0.0
    df['GrowthRate_Perim'] = 0.0
    
    area_col = 'Precise_Area' if 'Precise_Area' in df.columns else 'BBox_Area'
    perim_col = 'Precise_Perim' if 'Precise_Perim' in df.columns else 'BBox_Length' # Fallback
    
    for cycle in df['Cycle'].unique():
        cycle_mask = df['Cycle'] == cycle
        merged_mask = cycle_mask & df['IsMerged']
        merged_idx = df[merged_mask].index
        
        for i in range(1, len(merged_idx)):
            curr_idx = merged_idx[i]
            prev_idx = merged_idx[i-1]
            
            # Area Growth
            prev_area = df.loc[prev_idx, area_col]
            curr_area = df.loc[curr_idx, area_col]
            if prev_area > 0:
                df.loc[curr_idx, 'GrowthRate_Area'] = ((curr_area - prev_area) / prev_area) * 100.0
            
            # Perimeter Growth (Primary)
            prev_perim = df.loc[prev_idx, perim_col]
            curr_perim = df.loc[curr_idx, perim_col]
            if prev_perim > 0:
                df.loc[curr_idx, 'GrowthRate_Perim'] = ((curr_perim - prev_perim) / prev_perim) * 100.0
                
    return df

def remove_outliers_and_smooth(df):
    """
    Identifies and removes outliers in 'BBox_Length' and 'BBox_Breadth' per cycle.
    Uses rolling statistics (median and variance/std) to detect anomalies,
    replaces them, and interpolates to ensure smooth plotting.
    Then re-calculates Precise_Perim and other dependent metrics.
    Also applies a Savitzky-Golay smoothing filter to make trends clean.
    """
    if df.empty: return df
    
    cleaned_df = df.copy()
    
    # 1. Clean outliers in BBox_Length and BBox_Breadth for each cycle
    for cycle in cleaned_df['Cycle'].unique():
        idx = cleaned_df[cleaned_df['Cycle'] == cycle].index
        group = cleaned_df.loc[idx].sort_values('RelativeFrame')
        
        if len(group) < 5:
            continue
            
        for col in ['BBox_Length', 'BBox_Breadth']:
            # Using rolling median as local trend
            rolling_med = group[col].rolling(window=11, center=True, min_periods=1).median()
            
            # Variance of residuals (deviation from local trend)
            residuals = group[col] - rolling_med
            variance = residuals.var()
            std_dev = np.sqrt(variance) if variance > 0 else 1.0
            
            # Outlier threshold (e.g. 2 standard deviations, minimum of 3 pixels)
            threshold = max(2.0 * std_dev, 3.0)
            outliers = residuals.abs() > threshold
            
            cleaned_vals = group[col].copy()
            cleaned_vals[outliers] = np.nan
            
            # Interpolate to fill gaps smoothly
            cleaned_vals = cleaned_vals.interpolate(method='linear', limit_direction='both')
            cleaned_df.loc[idx, col] = cleaned_vals

    # 2. Recalculate BBox_Area based on cleaned values
    cleaned_df['BBox_Area'] = cleaned_df['BBox_Length'] * cleaned_df['BBox_Breadth']
    
    # 3. Update Precise_Perim using cleaned length and breadth
    # Identify if original perimeter was just BBox perimeter: 2 * (BBox_Length + BBox_Breadth)
    is_bbox_perim = (df['Precise_Perim'] - 2 * (df['BBox_Length'] + df['BBox_Breadth'])).abs() < 1e-3
    
    cleaned_df.loc[is_bbox_perim, 'Precise_Perim'] = 2 * (cleaned_df.loc[is_bbox_perim, 'BBox_Length'] + cleaned_df.loc[is_bbox_perim, 'BBox_Breadth'])
    
    # For mask-based perimeters (if any), also remove their outliers directly using the same method
    mask_perim_mask = ~is_bbox_perim
    if mask_perim_mask.any():
        for cycle in cleaned_df['Cycle'].unique():
            idx = cleaned_df[(cleaned_df['Cycle'] == cycle) & mask_perim_mask].index
            if len(idx) < 5: continue
            vals = cleaned_df.loc[idx, 'Precise_Perim']
            rolling_med = vals.rolling(window=11, center=True, min_periods=1).median()
            residuals = vals - rolling_med
            std_dev = np.sqrt(residuals.var()) if residuals.var() > 0 else 1.0
            threshold = max(2.0 * std_dev, 3.0)
            outliers = residuals.abs() > threshold
            cleaned_vals = vals.copy()
            cleaned_vals[outliers] = np.nan
            cleaned_vals = cleaned_vals.interpolate(method='linear', limit_direction='both')
            cleaned_df.loc[idx, 'Precise_Perim'] = cleaned_vals
            
    # 4. Apply Savitzky-Golay smoothing filter to length, breadth, and perimeter per cycle
    # to guarantee beautiful, smooth trends.
    from scipy.signal import savgol_filter
    for cycle in cleaned_df['Cycle'].unique():
        idx = cleaned_df[cleaned_df['Cycle'] == cycle].index
        group = cleaned_df.loc[idx].sort_values('RelativeFrame')
        if len(group) > 11:
            for col in ['BBox_Length', 'BBox_Breadth', 'Precise_Perim']:
                cleaned_df.loc[idx, col] = savgol_filter(group[col], window_length=11, polyorder=2)
                
    return cleaned_df

def plot_rog_data(df, output_path, title, metric='Precise_Perim'):
    """Generates the trend plots without dots (lines only), with bottom padding and proper cycle representations."""
    if df.empty: return
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    plt.xlabel('Relative Frame Number (Normalized)')
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.6)

    cycles = sorted(df['Cycle'].unique())
    for i, cycle in enumerate(cycles):
        c_df = df[df['Cycle'] == cycle].sort_values('RelativeFrame')
        if c_df.empty: continue
        color = colors[i % 20]
        
        pre = c_df[~c_df['IsMerged']]
        post = c_df[c_df['IsMerged']]
        
        label_added = False
        
        # Plot pre-merge line (no marker/dots, just dashed line)
        if not pre.empty:
            plt.plot(pre['RelativeFrame'], pre[metric], color=color, linestyle='--', alpha=0.6, 
                     label=f"Cycle {cycle}" if not label_added else "_nolegend_")
            label_added = True
            
        # Plot post-merge line (no marker/dots, just solid line)
        if not post.empty:
            plt.plot(post['RelativeFrame'], post[metric], color=color, linestyle='-', alpha=0.9, 
                     label=f"Cycle {cycle}" if not label_added else "_nolegend_")
            label_added = True
            
        if not pre.empty and not post.empty:
            merge_frame = post['RelativeFrame'].min()
            plt.axvline(x=merge_frame, color=color, linestyle=':', alpha=0.5)

    # Set y-axis limits to leave space below the final/lowest plotted curves
    y_min = df[metric].min()
    y_max = df[metric].max()
    y_range = y_max - y_min
    if y_range > 0:
        plt.ylim(y_min - 0.15 * y_range, y_max + 0.05 * y_range)
    else:
        plt.ylim(y_min - 5.0, y_max + 5.0)

    # Determine dynamic number of columns in the legend based on cycle count
    num_cycles = len(cycles)
    ncol = 1
    if num_cycles > 15:
        ncol = 3
    elif num_cycles > 8:
        ncol = 2

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='x-small', ncol=ncol)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # leaves space below the final plot
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_frame_analytics(df, analytics_dir):
    """Generates scatter plots for first/last merged frames without any area metrics."""
    if df.empty: return
    os.makedirs(analytics_dir, exist_ok=True)
    
    merged_df = df[df['IsMerged']]
    if merged_df.empty: return
    
    # 1. PER FRAME AVERAGES (No Area)
    metrics = {
        'BBox_Length': ['mean', 'min', 'max'],
        'BBox_Breadth': ['mean', 'min', 'max'],
        'Precise_Perim': ['mean', 'min', 'max']
    }
    
    avg_df = merged_df.groupby('RelativeFrame').agg(metrics)
    avg_df.columns = [f"{c[0]}_{c[1]}" for c in avg_df.columns]
    avg_df = avg_df.reset_index()
    avg_df.to_csv(os.path.join(analytics_dir, 'per_frame_averages.csv'), index=False)
    
    # 2. SCATTER DATA (No Area)
    scatter_data = []
    for c in merged_df['Cycle'].unique():
        c_df = merged_df[merged_df['Cycle'] == c].sort_values('RelativeFrame')
        if c_df.empty: continue
        scatter_data.append({
            'Cycle': c,
            'First_Frame': c_df.iloc[0]['RelativeFrame'],
            'First_Length': c_df.iloc[0]['BBox_Length'],
            'First_Breadth': c_df.iloc[0]['BBox_Breadth'],
            'First_Perim': c_df.iloc[0]['Precise_Perim'],
            'Last_Frame': c_df.iloc[-1]['RelativeFrame'],
            'Last_Length': c_df.iloc[-1]['BBox_Length'],
            'Last_Breadth': c_df.iloc[-1]['BBox_Breadth'],
            'Last_Perim': c_df.iloc[-1]['Precise_Perim'],
        })
    s_df = pd.DataFrame(scatter_data)
    
    def render_scatter(data, prefix, title, filename):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=16)
        metrics_to_plot = [
            (f'{prefix}_Length', 'Length (px)'),
            (f'{prefix}_Breadth', 'Breadth (px)'),
            (f'{prefix}_Perim', 'Perimeter (px)')
        ]
        for i, (m, label) in enumerate(metrics_to_plot):
            ax = axes[i]
            sc = ax.scatter(data[m], data[f'{prefix}_Frame'], c=data['Cycle'], cmap='viridis', s=80, edgecolors='k', alpha=0.8)
            ax.set_xlabel(label); ax.set_ylabel('Relative Frame Number')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Leave space below the lowest plotted points
            y_min = data[f'{prefix}_Frame'].min()
            y_max = data[f'{prefix}_Frame'].max()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(bottom=y_min - 0.15 * y_range)
            else:
                ax.set_ylim(bottom=y_min - 5.0)

        fig.colorbar(sc, ax=axes.tolist(), label='Cycle')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(analytics_dir, filename), dpi=200)
        plt.close()

    render_scatter(s_df, 'First', 'Metrics at First Merged Frame', 'first_frame_scatter.png')
    render_scatter(s_df, 'Last', 'Metrics at Last Merged Frame', 'last_frame_scatter.png')

def calculate_slopes(df, output_dir):
    """Calculates growth slopes per cycle."""
    if df.empty: return
    cycle_slopes = []
    for cycle, group in df.groupby('Cycle'):
        fit_data = group[group['IsMerged'] & (group['RelativeFrame'] >= 60) & (group['RelativeFrame'] <= 180)]
        if len(fit_data) < 5: continue
        slope_l, _ = np.polyfit(fit_data['RelativeFrame'], fit_data['BBox_Length'], 1)
        slope_b, _ = np.polyfit(fit_data['RelativeFrame'], fit_data['BBox_Breadth'], 1)
        cycle_slopes.append({'Cycle': cycle, 'Slope_L': slope_l, 'Slope_B': slope_b})
    pd.DataFrame(cycle_slopes).to_csv(os.path.join(output_dir, "bbox_slopes.csv"), index=False)
