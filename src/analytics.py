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

def plot_rog_data(df, output_path, title, metric='BBox_Area'):
    """Generates the trend plots."""
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
        
        if not pre.empty:
            plt.plot(pre['RelativeFrame'], pre[metric], color=color, linestyle='--', marker='^', markersize=4, alpha=0.6, label=f"Cycle {cycle} (Pre-merge)" if i < 10 else "_nolegend_")
        if not post.empty:
            plt.plot(post['RelativeFrame'], post[metric], color=color, linestyle='-', marker='o', markersize=4, alpha=0.9, label=f"Cycle {cycle} (Merged)" if i < 10 else "_nolegend_")
            
        if not pre.empty and not post.empty:
            merge_frame = post['RelativeFrame'].min()
            plt.axvline(x=merge_frame, color=color, linestyle=':', alpha=0.5)

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='x-small', ncol=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_frame_analytics(df, analytics_dir):
    """Generates scatter plots for first/last merged frames with shifted axes."""
    if df.empty: return
    os.makedirs(analytics_dir, exist_ok=True)
    
    merged_df = df[df['IsMerged']]
    if merged_df.empty: return
    
    area_col = 'Precise_Area' if 'Precise_Area' in merged_df.columns else 'BBox_Area'
    perim_col = 'Precise_Perim' if 'Precise_Perim' in merged_df.columns else 'BBox_Area'
    
    # 1. PER FRAME AVERAGES
    metrics = {
        'BBox_Length': ['mean', 'min', 'max'],
        'BBox_Breadth': ['mean', 'min', 'max'],
        area_col: ['mean', 'min', 'max'],
        'Precise_Perim': ['mean', 'min', 'max']
    }
    
    avg_df = merged_df.groupby('RelativeFrame').agg(metrics)
    avg_df.columns = [f"{c[0]}_{c[1]}" for c in avg_df.columns]
    avg_df = avg_df.reset_index()
    avg_df.to_csv(os.path.join(analytics_dir, 'per_frame_averages.csv'), index=False)
    
    # 2. SCATTER DATA
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
            'First_Area': c_df.iloc[0][area_col],
            'Last_Frame': c_df.iloc[-1]['RelativeFrame'],
            'Last_Length': c_df.iloc[-1]['BBox_Length'],
            'Last_Breadth': c_df.iloc[-1]['BBox_Breadth'],
            'Last_Perim': c_df.iloc[-1]['Precise_Perim'],
            'Last_Area': c_df.iloc[-1][area_col],
        })
    s_df = pd.DataFrame(scatter_data)
    
    def render_scatter(data, prefix, title, filename):
        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        fig.suptitle(title, fontsize=20)
        metrics_to_plot = [
            (f'{prefix}_Length', 'Length (px)'),
            (f'{prefix}_Breadth', 'Breadth (px)'),
            (f'{prefix}_Perim', 'Perimeter (px)'),
            (f'{prefix}_Area', 'Area (px^2)')
        ]
        for i, (m, label) in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            sc = ax.scatter(data[m], data[f'{prefix}_Frame'], c=data['Cycle'], cmap='viridis', s=80, edgecolors='k', alpha=0.8)
            ax.set_xlabel(label); ax.set_ylabel('Relative Frame Number')
            ax.grid(True, linestyle='--', alpha=0.6)
        fig.colorbar(sc, ax=axes.ravel().tolist(), label='Cycle')
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
