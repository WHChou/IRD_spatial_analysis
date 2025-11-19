from scipy.spatial import Delaunay, KDTree
from collections import Counter
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

def get_proximal_neighbors(df, x_col='x_centroid', y_col='y_centroid', distance_threshold=50):
    """
    Performs Delaunay triangulation, finds nearest neighbors in the Delaunay graph,
    and discards neighbors further than a threshold distance.

    Args:
        df (pd.DataFrame): DataFrame containing cell coordinates.
                           It's assumed that the DataFrame index is a simple range index
                           if you want to directly use the output dict keys to access df rows.
        x_col (str): Name of the column for x-coordinates.
        y_col (str): Name of the column for y-coordinates.
        distance_threshold (float): Maximum distance for a cell to be considered a neighbor.

    Returns:
        dict: A dictionary where keys are original DataFrame row indices
              and values are lists of original DataFrame row indices of their proximal neighbors.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")

    points = df[[x_col, y_col]].values
    original_indices_map = df.index # To map points array indices back to original df indices

    
    if len(points) < 3:
        # Delaunay triangulation requires at least 3 points to form a simplex (triangle in 2D).
        print("Not enough points to perform Delaunay triangulation. Need at least 3 points.")
        return {idx: [] for idx in df.index} # Return mapping to original DataFrame indices
        
    tri = Delaunay(points)

    # tri.vertex_neighbor_vertices is a CSR-like structure:
    # indptr[i] to indptr[i+1] gives the slice in 'indices' for the i-th point in the 'points' array
    # indices[indptr[i]:indptr[i+1]] gives the actual indices (referring to the 'points' array)
    # of the neighbors for the i-th point.
    indptr, delaunay_indices = tri.vertex_neighbor_vertices
    kdtree = KDTree(points)

    proximal_neighbors = {}
    for i in range(len(points)):
        current_original_idx = original_indices_map[i]
        
        # a. Get Delaunay neighbors (indices relative to 'points' array)
        delaunay_neighbors_for_i = delaunay_indices[indptr[i]:indptr[i+1]]
        delaunay_set = set(delaunay_neighbors_for_i)
        
        # b. Get K-D tree neighbors (indices relative to 'points' array)
        # query_ball_point returns a list of indices of points within the radius
        kdtree_neighbors_for_i = kdtree.query_ball_point(points[i], r=distance_threshold)
        # Remove the point itself from its k-d tree neighbors
        kdtree_set = set(kdtree_neighbors_for_i) - {i} 
        
        # c. Intersect the two sets
        intersected_neighbor_indices_in_points_array = list(delaunay_set.intersection(kdtree_set))
        
        # d. Map intersected indices back to original DataFrame indices
        final_neighbor_original_indices = [original_indices_map[n_idx] for n_idx in intersected_neighbor_indices_in_points_array]
        
        proximal_neighbors[current_original_idx] = final_neighbor_original_indices
        
    return proximal_neighbors


def neighboring_celltype_counts(cell_info, ct_col_name, ct_of_interest, neighbors_dict):
    # Count neighboring cell types of PS endothelial cells
    # cell_info: dataframe with cell centroid and cell type annotation
    # ct_col_name: name of the column with cell type annotation, e.g. 'group'
    # ct_of_interest: cell type of interest, e.g. 'PS Endothelial
    # neighbors_dict: output of get_proximal_neighbors function
    ct_index = cell_info[cell_info[ct_col_name] == ct_of_interest].index.values
    ct_neighbor_indices = [neighbors_dict[i] for i in ct_index]
    ct_neighbor_indices_flat = [item for sublist in ct_neighbor_indices for item in sublist]
    ct_neighbors = [cell_info[ct_col_name][i] for i in ct_neighbor_indices_flat]
    ct_neighbors_counter = Counter(ct_neighbors)
    
    # Get all unique cell types
    all_cell_types = np.unique(cell_info[ct_col_name])
    # Create dictionary with zeros for all cell types, then update with actual counts
    ct_neighbors_counts = {ct: 0 for ct in all_cell_types}
    ct_neighbors_counts.update(ct_neighbors_counter)

    return ct_neighbors_counts

def run_neighboring_celltype_analysis(df, ct_col, sample_col, distance_threshold = 25, x_col='x_centroid', y_col='y_centroid'):
    '''
    Calculates the neighboring cell type counts for each cell type in each sample.
    Args:
        df (pd.DataFrame): DataFrame containing cell information with columns for cell type, sample ID, and cell coordinates.
        ct_col (str): Name of the column containing cell type annotations.
        sample_col (str): Name of the column containing sample IDs.
        distance_threshold (float): Maximum distance for a cell to be considered a neighbor.
        x_col (str): Name of the column for x-coordinates.
        y_col (str): Name of the column for y-coordinates.
    Returns:
        pd.DataFrame: DataFrame with columns for cell type, neighboring cell type, sample ID, and counts of neighboring cell types.
    '''
    # Initialize an empty DataFrame to store results
    all_nb_df = pd.DataFrame()

    # Loop through each sample
    for sample in df[sample_col].unique().tolist():
        sample_df = df[df[sample_col] == sample]
        # Get proximal neighbor dict for all cells
        all_neighbors = get_proximal_neighbors(sample_df, x_col=x_col, y_col=y_col, distance_threshold=distance_threshold)

        for ct in sample_df[ct_col].unique().tolist():       
            # Count neighboring cell types of all cell types
            nb_counts = neighboring_celltype_counts(sample_df, ct_col, ct, all_neighbors)
            nb_df = pd.DataFrame.from_dict(nb_counts, orient = "index", columns = ["counts"])
            nb_df['Cell Type'] = ct
            nb_df['Neighboring Cell Type'] = nb_df.index
            nb_df['Sample'] = sample
            nb_df.reset_index(drop = True, inplace = True)
            all_nb_df = pd.concat([all_nb_df, nb_df], axis = 0)
            
        print(f"Finished processing sample {sample}")
    
    return all_nb_df

def nearest_dist_to_ref_celltype(df, ct_col_name, ref_ct, x_col='x_centroid', y_col='y_centroid'):
    """
    This function finds the nearest distance for all cells in df to the nearest cell of a reference cell type.

    Args:
        df (pd.DataFrame): DataFrame containing cell coordinates and cell type annotation.
        ct_col_name (str): Name of the column in df that contains cell type annotations.
        ref_ct (str): The reference cell type to which distances will be calculated.
        x_col (str): Name of the column for x-coordinates. Default is 'x_centroid'.
        y_col (str): Name of the column for y-coordinates. Default is 'y_centroid'.
    Returns:
        res_df (pd.DataFrame): DataFrame with original df columns plus a new column 'nearest_dist_to_{ref_ct}'.
    """
    # Check inputs

    # Extract coordinates of reference cell type to build kdtree
    ref_cells = df[df[ct_col_name] == ref_ct]
    if ref_cells.empty:
        raise ValueError(f"No cells found for reference cell type '{ref_ct}' in column '{ct_col_name}'.")
    ref_tree = KDTree(ref_cells[[x_col, y_col]].values)

    # Query kdtree for all cells to find nearest reference cell type
    all_cells_coords = df[[x_col, y_col]].values
    dists, _ = ref_tree.query(all_cells_coords)

    # Create result DataFrame
    res_df = df.copy()
    res_df[f'nearest_dist_to_{ref_ct}'] = dists

    return res_df


def nearest_dist_to_ref_celltype_allsamples(df, ref_ct, ct_col, sample_col, dist_eval, x_col='x_centroid', y_col='y_centroid'):
    ''' 
    This function calculates the most likely distance to a reference cell type for all other cell types, across all samples.
    Args:
        df (pd.DataFrame): DataFrame containing cell coordinates, cell type annotation, and sample information.
        ref_ct (str): The reference cell type to which distances will be calculated.
        ct_col (str): Name of the column in df that contains cell type annotations.
        sample_col (str): Name of the column in df that contains sample identifiers.
        dist_eval (np.array): Array of distance values over which to evaluate the KDE.
        x_col (str): Name of the column for x-coordinates. Default is 'x_centroid'.
        y_col (str): Name of the column for y-coordinates. Default is 'y_centroid'.
    Returns:
        dist_to_ref_df (pd.DataFrame): DataFrame with columns ['Sample', 'Cell Type', 'Most Likely Distance'].
    '''
    # Initialize empty dataframe to store results
    dist_to_ref_df = pd.DataFrame(columns = ['Sample', 'Cell Type', 'Most Likely Distance'])

    # Identify cell types to evaluate, excluding the reference cell type
    all_other_ct = df[ct_col].unique().tolist()
    all_other_ct.remove(ref_ct)

    # Loop through all samples
    nSamples = len(df[sample_col].unique().tolist())
    for i in range(nSamples):
        sample = df[sample_col].unique().tolist()[i]
        sample_df = df[df[sample_col] == sample]
        if sample_df[sample_df[ct_col] == ref_ct].empty:
            print(f"Skipping sample {sample} as it has no cells of reference cell type '{ref_ct}'")
            continue
        sample_df1 = nearest_dist_to_ref_celltype(sample_df, ct_col, ref_ct, x_col='x_centroid', y_col='y_centroid')
        
        dist = []
        for ct in all_other_ct:
            dist_distribution = sample_df1[sample_df1[ct_col]==ct][f'nearest_dist_to_{ref_ct}']
            if len(dist_distribution) < 10: # Require at least 10 data points to estimate KDE
                dist.append(np.nan)
                continue
            else:
                kde_ct = gaussian_kde(dist_distribution)
                ct_kde_pdf = kde_ct.pdf(dist_eval)
                ct_dist = dist_eval[np.argmax(ct_kde_pdf)]
                dist.append(ct_dist)

        dist_to_ref_ct_df = pd.DataFrame({
            'Sample': sample,
            'Cell Type': all_other_ct,
            'Most Likely Distance': dist
        })
        dist_to_ref_df = pd.concat([dist_to_ref_df, dist_to_ref_ct_df], axis = 0)
    return dist_to_ref_df

def nearest_dist_between_two_celltypes(df1, df2, sample_col, x_col='x_centroid', y_col='y_centroid'):
    """
    This function finds the nearest distance for all cells in df1 to the nearest cell in df2, within each sample.

    Args:
        df1 (pd.DataFrame): DataFrame containing cell coordinates and sample information for the first cell type.
        df2 (pd.DataFrame): DataFrame containing cell coordinates and sample information for the second cell type.
        sample_col (str): Name of the column in df1 and df2 that contains sample identifiers. If set to None, all cells are treated as from the same sample.
        x_col (str): Name of the column for x-coordinates. Default is 'x_centroid'.
        y_col (str): Name of the column for y-coordinates. Default is 'y_centroid'.
    Returns:
        res_df (pd.DataFrame): DataFrame with original df1 columns plus a new column 'nearest_dist_to_df2'.
    """
    # Check inputs

    # Initialize empty list to store results
    results = []
    # If sample_col is None, create a dummy sample column
    if sample_col is None:
        df1[sample_col] = 'all_samples'
        df2[sample_col] = 'all_samples'
    # Loop through each sample
    all_samples = np.union1d(df1[sample_col].unique(), df2[sample_col].unique())
    for sample in all_samples:
        df1_sample = df1[df1[sample_col] == sample]
        df2_sample = df2[df2[sample_col] == sample]

        if df1_sample.empty:
            print(f"Skipping sample {sample} as it has no cells in df1.")
            continue

        if df2_sample.empty:
            print(f"Skipping sample {sample} as it has no cells in df2.")
            continue

        # Build KDTree for df2
        df2_tree = KDTree(df2_sample[[x_col, y_col]].values)

        # Query KDTree for all cells in df1 to find nearest cell in df2
        dists, _ = df2_tree.query(df1_sample[[x_col, y_col]].values)

        # Create result DataFrame for this sample
        res_sample_df = df1_sample.copy()
        res_sample_df['nearest_dist_to_df2'] = dists

        results.append(res_sample_df)

    # Concatenate all results into a single DataFrame
    res_df = pd.concat(results, axis=0).reset_index(drop=True)

    return res_df

def compute_pairwise_LR_scores(adata, ligand_gene, receptor_gene,
                               distance_threshold=200, dist_lambda = 50,
                               expr_layer = None, celltype_col='annot', x_col='x_centroid', y_col='y_centroid'):
    """
    DOCS (This is per sample)
    """
    # 1) Get coordinates and expression
    coords = adata.obs[[x_col, y_col]].copy().values
    if expr_layer is None:
        L_exp = adata[:, ligand_gene].X.toarray().flatten()
        R_exp = adata[:, receptor_gene].X.toarray().flatten()
    else:
        L_exp = adata[:, ligand_gene].layers[expr_layer].toarray().flatten()
        R_exp = adata[:, receptor_gene].layers[expr_layer].toarray().flatten()

    # Find cells expressing ligand and receptor
    ligand_idx = np.where(L_exp > 0)[0]
    receptor_idx = np.where(R_exp > 0)[0]

    # If no ligand or receptor expressing cells, return empty dataframe
    if len(ligand_idx) == 0 or len(receptor_idx) == 0:
        print(f"No ligand or receptor expressing cells for {ligand_gene}-{receptor_gene} in this sample.")
        return pd.DataFrame(), pd.DataFrame()

    # 2) Compute ligand receptor pairs within distance threshold
    # Build KDTree for receptor cells
    tree = KDTree(coords[receptor_idx])
    # For all ligand points, find all points within distance threshold
    results = tree.query_ball_point(coords[ligand_idx], r=distance_threshold)

    # 3) Compute scores for each ligand-receptor pair
    score_results = []
    for li_idx, neighbors in enumerate(results):
        ligand_cell_idx = ligand_idx[li_idx]
        if len(neighbors) == 0:
            continue
        # Vectorized distance calculation
        neighbor_global_idx = receptor_idx[neighbors]
        dists = np.linalg.norm(coords[ligand_cell_idx] - coords[neighbor_global_idx], axis=1)
        Li = L_exp[ligand_cell_idx]
        Rj = R_exp[neighbor_global_idx]
        K = np.exp(-dists / dist_lambda)
        score_ij = Li * Rj * K
        # Store results
        for r_idx, dist, L_exp_val, R_exp_val, score in zip(neighbor_global_idx, dists, 
                                                        [Li]*len(neighbor_global_idx), 
                                                        Rj, score_ij):
            score_results.append({
                'Ligand Cell Index': ligand_cell_idx,
                'Receptor Cell Index': r_idx,
                'Ligand Cell Type': adata.obs[celltype_col].iloc[ligand_cell_idx],
                'Receptor Cell Type': adata.obs[celltype_col].iloc[r_idx],
                'Distance': dist,
                'Ligand Expression': L_exp_val,
                'Receptor Expression': R_exp_val,
                'LR Score': score
            })

    pairwise_scores_df = pd.DataFrame(score_results)
    
    # aggregation per celltype pair
    agg = pairwise_scores_df.groupby(['Ligand Cell Type','Receptor Cell Type']).agg(
        score_sum=('LR Score','sum'),
        score_mean=('LR Score','mean'),
        pair_count=('LR Score','count')
    ).reset_index()

    # normalize by possible pairs (N_i * N_j)
    counts = adata.obs[celltype_col].value_counts().to_dict()
    agg['N_ligandCt'] = agg['Ligand Cell Type'].map(counts)
    agg['N_ReceptorCt'] = agg['Receptor Cell Type'].map(counts)
    agg['norm_score'] = agg['score_sum'] / (agg['N_ligandCt'] * agg['N_ReceptorCt'])

    return pairwise_scores_df, agg

def compute_LR_scores(adata, ligand_gene, receptor_gene,
                      distance_threshold=200, dist_lambda = 50,
                      expr_layer = None, celltype_col='annot', sample_col='Sample',
                      x_col='x_centroid', y_col='y_centroid'):
    """
    DOCS
    """
    all_scores = []
    all_agg = []
    for sample in adata.obs[sample_col].unique():
        print(f"Processing sample: {sample}")
        sample_adata = adata[adata.obs[sample_col] == sample]
        sample_scores, agg = compute_pairwise_LR_scores(sample_adata, ligand_gene, receptor_gene,
                                                   distance_threshold, dist_lambda,
                                                   expr_layer, celltype_col, x_col, y_col)
        sample_scores['Sample'] = sample
        agg['Sample'] = sample
        all_agg.append(agg)
        all_scores.append(sample_scores)

    all_scores_df = pd.concat(all_scores, axis=0)
    all_agg_df = pd.concat(all_agg, axis=0)
    
    return all_scores_df, all_agg_df