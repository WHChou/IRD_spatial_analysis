import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparison_with_significance(data, x_col, y_col, order=None, palette='Set2', 
                                       xlabel=None, ylabel=None, title=None,
                                       save_path=None, figsize=(6, 5)):
    """
    Create boxplot with swarmplot overlay and significance brackets between groups.
    
    Automatically performs pairwise Mann-Whitney U tests and displays significance
    brackets above the plot with stacked layout for multiple comparisons.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data frame with measurements per sample/group
    x_col : str
        Column name for x-axis groups (e.g., 'Collection')
    y_col : str
        Column name for y-axis values
    order : list, optional
        Order of x-axis categories
    palette : str
        Seaborn color palette
    xlabel, ylabel, title : str, optional
        Axis labels and title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns:
    --------
    tuple: (matplotlib.figure.Figure, pd.DataFrame)
        Figure object and dataframe of statistics results
        
    Example:
    --------
    fig, stats = plot_comparison_with_significance(
        myeloid_supp_pct, 
        x_col='Collection', 
        y_col='PDL1_pos',
        order=['NBM', 'NDMM', 'PT'],
        ylabel='Fraction of PDL1+ myeloid cells'
    )
    """
    
    # Step 1: Calculate statistics between all pairs
    stats_res = []
    groups = data[x_col].unique() if order is None else order
    
    for cond1, cond2 in combinations(groups, 2):
        vals1 = data.loc[data[x_col] == cond1, y_col]
        vals2 = data.loc[data[x_col] == cond2, y_col]
        if len(vals1) > 0 and len(vals2) > 0:
            pval = mannwhitneyu(vals1, vals2).pvalue
            stats_res.append({'Group1': cond1, 'Group2': cond2, 'pval': pval})
    
    stats_df = pd.DataFrame(stats_res)
    
    # Step 2: Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, x=x_col, y=y_col, order=order, palette=palette, ax=ax)
    sns.swarmplot(data=data, x=x_col, y=y_col, order=order, color='black', size=6, alpha=0.5, ax=ax)
    
    # Step 3: Add significance brackets
    if len(stats_df) > 0:
        x_pos = {lab: i for i, lab in enumerate(groups)}
        
        ymin, ymax = ax.get_ylim()
        y_span = ymax - ymin
        pad = 0.05 * y_span
        step = 0.08 * y_span
        h = 0.015 * y_span
        
        base = data[y_col].max() + pad
        levels = [0] * len(groups)
        
        # Sort pairs by span (shorter first for better layout)
        pairs = []
        for _, r in stats_df.iterrows():
            i, j = x_pos[r['Group1']], x_pos[r['Group2']]
            if i > j:
                i, j = j, i
            pairs.append((i, j, r['pval']))
        pairs.sort(key=lambda t: (t[1] - t[0], t[0]))
        
        # Draw brackets
        for i, j, p in pairs:
            lvl = max(levels[i:j+1])
            y1 = base + lvl * step
            y0 = y1 - h
            
            ax.plot([i, i, j, j], [y0, y1, y1, y0], lw=1.5, c='k', clip_on=False)
            ax.text((i + j) / 2, y1 + 0.5*h, f"p={p:.2g}", 
                   ha='center', va='bottom', fontsize=9, clip_on=False)
            
            for k in range(i, j + 1):
                levels[k] = lvl + 1
        
        # Adjust ylim for brackets
        needed_top = base + (max(levels) * step) + pad
        if needed_top > ymax:
            ax.set_ylim(ymin, needed_top)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig, stats_df

def plot_multigroup_boxplot_with_significance(df, x_col, y_col, hue_col, 
                                              order=None, hue_order=None, show_outliers = True,show_swarm=True, 
                                              palette = 'Set2', figsize=(6, 5), xlabel=None, ylabel=None, title=None, save_path=None):
    # Step 1: Calculate statistical test results
    stats_res = []
    hues = df[hue_col].unique().tolist()

    for groups in df[x_col].unique():
        sub_df = df[df[x_col] == groups]

        for cond1, cond2 in combinations(hues, 2):
            cond1_df = sub_df[sub_df[hue_col] == cond1][y_col].values.tolist()
            cond2_df = sub_df[sub_df[hue_col] == cond2][y_col].values.tolist()
            if len(cond1_df) > 0 and len(cond2_df) > 0:
                u_stat, p_val = mannwhitneyu(cond1_df, cond2_df, alternative='two-sided')
                stats_res.append({'Condition 1': cond1, 'Condition 2': cond2, 'Group': groups, 'U statistic': u_stat, 'p-value': p_val})
    stats_df = pd.DataFrame(stats_res)
    stats_df['p_adj'] = multipletests(stats_df['p-value'], method='fdr_bh')[1]
    stats_df['Sig'] = stats_df['p_adj'] < 0.05
    
    # Step 2: Plot boxplot and swarmplot
    if order is None:
        order = df[x_col].unique().tolist()          # set x order
    if hue_order is None:
        hue_order = sorted(df[hue_col].unique())     # set hue order

    fig, ax = plt.subplots(figsize=figsize)                                      
    sns.boxplot(df, x = x_col, y = y_col, hue = hue_col, order=order, hue_order=hue_order, palette=palette, showfliers = show_outliers)
    if show_swarm:
        sns.swarmplot(df, x = x_col, y = y_col, hue = hue_col, order=order, hue_order=hue_order, dodge = True, color = 'k', alpha = 0.5, size = 3.5)
                                              
    # Step 3: Draw significance bars, avoiding overlap
    # Geometry for stacked annotations
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin
    pad = 0.04 * y_span      # space above boxes
    step = 0.08 * y_span     # vertical spacing between annotations
    h = 0.015 * y_span       # bracket height
    
    top_y_by_group = df.groupby(x_col)[y_col].max().to_dict()
    top_y_by_group = {k: v + pad for k, v in top_y_by_group.items()}
    
    stack_idx = {k: 0 for k in order}
    
    # Avoid duplicate pair annotations within the same group
    seen_pairs = set()
    
    for _, row in stats_df.iterrows():
        group = row['Group']
        cond1, cond2 = row['Condition 1'], row['Condition 2']
    
        # de-duplicate symmetrical pairs per neighborhood
        key = (group, tuple(sorted([cond1, cond2])))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
    
        i = order.index(group)
        j1 = hue_order.index(cond1)
        j2 = hue_order.index(cond2)
        x1 = dodge_center(i, j1, len(hue_order), width=0.8)
        x2 = dodge_center(i, j2, len(hue_order), width=0.8)
    
        # stacked y position
        y_base = top_y_by_group[group] + stack_idx[group] * step
        y0, y1 = y_base - h, y_base
    
        ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], lw=1.5, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, y1 + 0.5*h, f"p={row['p_adj']:.2g}",
                ha='center', va='bottom', color='k', fontsize=8, clip_on=False)
    
        stack_idx[group] += 1
    
    # Ensure enough headroom
    max_top = max((top_y_by_group[k] + max(0, stack_idx[k]-1) * step + pad) for k in order)
    if max_top > ymax:
        ax.set_ylim(ymin, max_top)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)

    return fig, stats_df

def dodge_center(i, j, n_hue, width=0.8):
        # seaborn's default group width is ~0.8
        if n_hue <= 1:
            return i
        step = width / n_hue
        return i - width/2 + step/2 + j*step