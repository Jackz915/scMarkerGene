import numpy as np
import pandas as pd
import copy
import os,sys,datetime,time,gzip
import math
import random
import argparse
import itertools
from sklearn.metrics import silhouette_score
from scipy import stats
from statsmodels.stats.multitest import multipletests




# print information line
def infoLine(message, infoType="info"):
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType=infoType + " "
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline = "[" + infoType + " " + str(time) + "] " + message
    print(outline)


def readGeneList(inDIR):
    """Read ordered gene list"""
    geneList = []
    with open(os.path.join(inDIR, "001_prepareData_output", "orderedGeneList.dat"), "rt") as fi:
        for line in fi:
            geneList.append(line.rstrip())
    return geneList

def readCellSample(inDIR):
    """Read cell ID list"""
    cellIdList = []
    with open(inDIR + "/001_prepareData_output/sampleList.pool.dat", "rt") as fi:
        for line in fi:
            cellIdList.append(line.rstrip())
    return cellIdList

def readPredictionInfo(inDIR, model_index):
    """Read model prediction information"""
    predList = []
    with open(inDIR + f"/002_geneContribution_output/predict_dir/model_{model_index}_pool.tab", "rt") as fi:
        f1st = True
        for line in fi:
            if f1st:
                f1st = False
                continue
            row = line.rstrip().split("\t")
            predList.append(row[0] == row[1])  # Whether prediction is correct
    return predList

def readCellCode(inDIR):
    """Read cell group coding"""
    codeHash = {}
    with open(inDIR + "/001_prepareData_output/cellCode.dat", "rt") as fi:
        for line in fi:
            row = line.rstrip().split("\t")
            codeHash[row[0]] = row[1]
    return codeHash

def readAggregatedExplanation(agg_file, geneList):
    """
    Read aggregated explanation file and return DataFrame
    
    Parameters:
    -----------
    agg_file : str
        Path to the aggregated explanation file
    geneList : list
        List of gene names (can be filtered subset)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: cell_id, group, and gene columns from geneList
    """
    data = []
    with open(agg_file, "rt") as fi:
        for line in fi:
            row = line.rstrip().split("\t")
            if len(row) >= 3:
                cell_id = row[0]
                group = row[1]
                att_vals = [float(x) for x in row[2].split("|")]
                att_vals = att_vals[:len(geneList)]
                gene_att = dict(zip(geneList, att_vals))
                data.append({"cell_id": cell_id, "group": group, **gene_att})
    
    df = pd.DataFrame(data)
    return df

def convert_df_to_matrix(df_agg, gene_list_dict=None, stat="median", exp=False):
    """
    Convert aggregated DataFrame to group-wise statistics matrix
    """
    gene_cols = [col for col in df_agg.columns if col not in ["cell_id", "group"]]
    
    if stat == "median":
        df_group = df_agg.groupby("group")[gene_cols].median()
    elif stat == "mean":
        df_group = df_agg.groupby("group")[gene_cols].mean()
    else:
        raise ValueError(f"Unsupported statistic: {stat}")
    
    return df_group.T
   

def get_significant_gene_names(gene_stats):
    return (
        gene_stats
        .loc[gene_stats["significant"], "gene"]
        .unique()
        .tolist()
    )


# ========== Data Collection Functions ==========
def collect_contribution_data(inDIR):
    """
    Collect contribution data from all models
    Returns:
      gene_model_arrays,
      cell_model_data,
      meta_info
    """

    codeHash = readCellCode(inDIR)
    cellIdList = readCellSample(inDIR)
    gene_names = readGeneList(inDIR)

    model_files = [
        f for f in os.listdir(inDIR + "/002_geneContribution_output/explain_dir/")
        if f.endswith(".dat")
    ]
    model_count = len(model_files)
    n_genes = len(gene_names)

    gene_model_data = [[] for _ in range(n_genes)]
    cell_model_data = {}
    cellId2groupHash = {}

    valid_cell_ids = set()

    for model_index in range(model_count):
        predList = readPredictionInfo(inDIR, model_index)

        with open(
            inDIR + f"/002_geneContribution_output/explain_dir/model_{model_index}_explanation.dat",
            "rt"
        ) as fi:

            for cell_index, line in enumerate(fi):
                cellId = cellIdList[cell_index]
                row = line.rstrip().split("\t")

                if not predList[cell_index]:
                    continue

                contributions = np.array([float(k) for k in row[1].split("|")])

                valid_cell_ids.add(cellId)

                # gene-level
                for gene_idx in range(n_genes):
                    if len(gene_model_data[gene_idx]) <= model_index:
                        gene_model_data[gene_idx].append([])
                    gene_model_data[gene_idx][model_index].append(contributions[gene_idx])

                # cell-level
                if cellId not in cell_model_data:
                    cell_model_data[cellId] = []
                cell_model_data[cellId].append(contributions)

                if cellId not in cellId2groupHash:
                    cellId2groupHash[cellId] = codeHash.get(row[0], "unknown")

    gene_model_arrays = []
    for gene_idx in range(n_genes):
        gene_model_arrays.append([
            np.array(m) for m in gene_model_data[gene_idx]
        ])

    meta_info = {
        "n_genes": n_genes,
        "n_models": model_count,
        "gene_names": gene_names,
        "cellId2groupHash": cellId2groupHash,
        "valid_cell_ids": sorted(valid_cell_ids)
    }

    return gene_model_arrays, cell_model_data, meta_info

def build_cell_gene_matrix(cell_model_data, gene_names):
    """
    Build cell x gene contribution matrix by averaging over models
    """
    cell_ids = list(cell_model_data.keys())
    n_cells = len(cell_ids)
    n_genes = len(gene_names)

    X = np.zeros((n_cells, n_genes))

    for i, cellId in enumerate(cell_ids):
        # cell_model_data[cellId]: list of (n_genes,) arrays
        contribs = np.array(cell_model_data[cellId])   # (n_models, n_genes)
        X[i] = contribs.mean(axis=0)                    # average over models

    return X, cell_ids




def compute_fdr_by_permutation(
    X,
    cell_ids,
    cellId2groupHash,
    gene_names,
    n_perm=100,
    min_cells=5,
    random_state=0
):
    """
    Global gene-level permutation test:
    H0: gene contribution score is independent of cell type
    """

    rng = np.random.default_rng(random_state)

    # ===== labels =====
    labels = np.array([cellId2groupHash[cid] for cid in cell_ids])
    cell_types, label_idx = np.unique(labels, return_inverse=True)

    n_cells, n_genes = X.shape
    n_types = len(cell_types)

    # ===== filter rare cell types =====
    valid_types = [
        i for i in range(n_types)
        if np.sum(label_idx == i) >= min_cells
    ]
    mask_valid = np.isin(label_idx, valid_types)

    X = X[mask_valid]
    label_idx = label_idx[mask_valid]

    # ===== helper: compute T_g =====
    def compute_T(X, labels):
        """Variance of cell-type means per gene"""
        means = np.zeros((len(valid_types), X.shape[1]))
        for i, ct in enumerate(valid_types):
            means[i] = np.nanmean(X[labels == ct], axis=0)
        return np.nanvar(means, axis=0)

    # ===== observed statistic =====
    T_obs = compute_T(X, label_idx)  # (n_genes,)

    if n_perm is None:
        df = pd.DataFrame({
            "gene": gene_names,
            "T_stat": T_obs,
            "p_value": np.nan,
            "fdr": np.nan,
            "significant": True
        }).sort_values("T_stat", ascending=False)
        return df
        
    # ===== permutation =====
    T_perm = np.zeros((n_perm, n_genes), dtype=np.float32)

    for b in range(n_perm):
        perm_labels = rng.permutation(label_idx)
        T_perm[b] = compute_T(X, perm_labels)

    # ===== empirical p-values =====
    pvals = (np.sum(T_perm >= T_obs, axis=0) + 1) / (n_perm + 1)

    # ===== FDR =====
    rejected, fdr, _, _ = multipletests(pvals, method="fdr_bh")

    # ===== output =====
    df = pd.DataFrame({
        "gene": gene_names,
        "T_stat": T_obs,
        "p_value": pvals,
        "fdr": fdr,
        "significant": rejected
    }).sort_values("p_value")

    return df



def save_statistical_results(gene_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gene_stats.to_csv(
        f"{output_dir}/gene_statistics.csv",
        index=False,
        float_format="%.6f"
    )
    
def save_aggregated_contributions(cell_model_data, cellId2groupHash, output_dir):
    """Save aggregated contribution data"""
    aggregated = {}
    for cellId, model_contribs in cell_model_data.items():
        if model_contribs:  # Ensure there is data
            aggregated[cellId] = np.mean(model_contribs, axis=0)
    
    with open(f"{output_dir}/aggregated_explanation.dat", "wt") as fo:
        for cellId, contrib in aggregated.items():
            group = cellId2groupHash.get(cellId, "unknown")
            att_str = "|".join([f"{k:.6f}" for k in contrib])
            fo.write(f"{cellId}\t{group}\t{att_str}\n")
    
    return aggregated


def aggregateModel(inDIR, outDIR, n_perm):
    """
    Main function: Aggregate model results and compute statistical significance
    """
    infoLine("Collecting contribution data from all models")
    gene_model_arrays, cell_model_data, meta_info = collect_contribution_data(inDIR)
    
    infoLine(f"Computing statistical significance for {len(gene_model_arrays)} genes")
    X, cell_ids = build_cell_gene_matrix(
        cell_model_data,
        meta_info["gene_names"]
    )
    
    gene_stats = compute_fdr_by_permutation(
        X=X,
        cell_ids=cell_ids,
        cellId2groupHash=meta_info["cellId2groupHash"],
        gene_names=meta_info["gene_names"],
        n_perm=n_perm
    )
        
    infoLine("Saving statistical results")
    output_dir = f"{outDIR}/aggregatedData"
    save_statistical_results(gene_stats, output_dir)
    
    infoLine("Saving aggregated contributions")
    aggregated = save_aggregated_contributions(
        cell_model_data, 
        meta_info['cellId2groupHash'], 
        output_dir
    )
    
    # Get significant gene names
    significant_gene_names = gene_stats.loc[gene_stats["significant"], "gene"].unique().tolist()
    infoLine(f"Analysis completed! Found {len(significant_gene_names)} significant genes (FDR<0.05) out of {meta_info['n_genes']} total genes")
    
    return significant_gene_names


#Candidate gene selection based on contribution scores
def extractPotentialGenes(inDIR, outDIR, TopN, significant_gene_names=None):
    """
    Extract potential marker genes based on contribution scores.
    
    Parameters:
    -----------
    inDIR : str
        Input directory containing data files
    outDIR : str
        Output directory containing aggregated results
    TopN : int
        Number of top candidate genes to select per group
    significant_gene_names : list, optional
        List of significant gene names to filter candidates.
        If None, use all genes from gene list.
    
    Returns:
    --------
    marker_gene_dict : dict
        Dictionary mapping group -> list of candidate genes
    """
    
    # ---------- Select Candidate genes ----------
    def select_top_contribution_markers(
        df_agg,
        df_median,
        df_mean,
        significant_gene_names=None,
        top_n=20,
        max_ratio_cutoffs=[0.1, 0.2, 0.3, 0.5],
        nonzero_cutoffs=[0.5, 0.3, 0.1, 0.0]
    ):
        """
        Select the top candidate genes for each group based on contribution scores
        and non-zero expression fraction across groups.
    
        Parameters:
        -----------
        df_agg : pd.DataFrame
            Aggregated contribution score DataFrame with columns ['cell_id', 'group', gene1, gene2, ...]
        df_median : pd.DataFrame
            Per-group gene median contribution score DataFrame (genes x groups)
        df_mean : pd.DataFrame
            Per-group gene mean contribution score DataFrame (genes x groups)
        significant_gene_names : list, optional
            List of significant gene names to filter candidates. If None, use all genes.
        top_n : int
            Number of top marker genes to select per group
        max_ratio_cutoffs : list
            Fold-change thresholds to compare target vs. other groups
        nonzero_cutoffs : list
            Minimum non-zero contribution score thresholds (progressively relaxed)
    
        Returns:
        --------
        marker_gene_dict : dict
            Dictionary mapping group -> list of candidate genes
        """
    
        # Calculate non-zero ratio
        def compute_nonzero_fraction(df_long):
            df_long["nonzero"] = df_long["value"] > 0
            return df_long.groupby(["gene", "group"])["nonzero"].mean().unstack()
    
        # Extract gene columns
        gene_cols = [col for col in df_agg.columns if col not in ["cell_id", "group"]]
        
        df_long = df_agg.melt(id_vars=["cell_id", "group"], value_vars=gene_cols,
                              var_name="gene", value_name="value")
    
        # Extract the genes with non-zero contribution scores
        nonzero_frac_df = compute_nonzero_fraction(df_long)
    
        marker_gene_dict = {}
    
        for group in df_median.columns:
            median_vals = df_median[group]
            mean_vals = df_mean[group]
    
            # Extract genes with non-zero contribution scores in at least one cell type
            # Filter by significant genes if provided
            candidate_genes = list(set(median_vals[median_vals > 0].index) &
                                   set(mean_vals[mean_vals > 0].index))
            
            selected = []  # Store selected markers
    
            # Try progressively relaxed fold-change cutoffs
            for ratio_cutoff in max_ratio_cutoffs:
                both_pass = []
                only_one_pass = []
    
                for gene in candidate_genes:
                    mv = median_vals[gene]
                    mm = mean_vals[gene]
                    mv_others = df_median.loc[gene, df_median.columns != group]
                    mm_others = df_mean.loc[gene, df_mean.columns != group]
    
                    # Check non-zero fraction requirement
                    if gene in nonzero_frac_df.index and group in nonzero_frac_df.columns:
                        nonzero_pass = nonzero_frac_df.loc[gene, group] >= nonzero_cutoffs[0]
                    else:
                        nonzero_pass = False
    
                    # Check fold-change across other groups
                    pass_median = nonzero_pass and all(val < mv * ratio_cutoff for val in mv_others)
                    pass_mean   = nonzero_pass and all(val < mm * ratio_cutoff for val in mm_others)
    
                    if pass_median and pass_mean:
                        both_pass.append((gene, mv, mm))
                    elif pass_median or pass_mean:
                        only_one_pass.append((gene, mv, mm))
    
                # Sort and select top genes passing both criteria
                both_pass = sorted(both_pass, key=lambda x: (-x[1], -x[2]))
                selected = both_pass[:top_n]
    
                # Fill in from one-pass genes if needed
                if len(selected) < top_n:
                    already = set(g[0] for g in selected)
                    only_one_pass = [g for g in only_one_pass if g[0] not in already]
                    only_one_pass = sorted(only_one_pass, key=lambda x: (-x[1], -x[2]))
                    selected.extend(only_one_pass[:top_n - len(selected)])
    
                if len(selected) >= top_n:
                    break  # Stop if we have enough genes
    
            # If still not enough genes, try relaxing non-zero cutoff
            if len(selected) < top_n:
                already = set(g[0] for g in selected)
                for cutoff in nonzero_cutoffs[1:]:
                    extras = []
                    for gene in candidate_genes:
                        if gene in already:
                            continue
                        if gene in nonzero_frac_df.index and group in nonzero_frac_df.columns:
                            if nonzero_frac_df.loc[gene, group] >= cutoff:
                                extras.append((gene, median_vals[gene], mean_vals[gene]))
                    extras = sorted(extras, key=lambda x: (-x[1], -x[2]))
                    selected.extend(extras[:top_n - len(selected)])
                    if len(selected) >= top_n:
                        break
                        
            # Warn if not enough genes found
            if len(selected) < top_n:
                print(f"[{group}] Only selected {len(selected)} candidate genes (less than {top_n})")

            # === ONLY HERE apply significant_gene_names as output mask ===
            selected_genes = [g[0] for g in selected]
            
            if significant_gene_names is not None:
                selected_genes = [g for g in selected_genes if g in significant_gene_names]
            
            marker_gene_dict[group] = selected_genes[:top_n]
    
        return marker_gene_dict

    # ---------- Pipeline ----------
    infoLine("Reading gene list")
    geneList = readGeneList(inDIR)

    infoLine("Contribution score calculation: median and mean")
    agg_file = os.path.join(outDIR, "aggregatedData", "aggregated_explanation.dat")
    
    # Check if aggregated file exists
    if not os.path.exists(agg_file):
        raise FileNotFoundError(f"Aggregated explanation file not found: {agg_file}")
    
    df_agg = readAggregatedExplanation(agg_file, geneList)

    # If dataframe is empty, return empty dict
    if df_agg.empty:
        print("Warning: No data in aggregated explanation file")
        return {}

    df_median_cs_all = convert_df_to_matrix(df_agg, gene_list_dict=None, stat="median", exp=False).clip(lower=0)
    df_mean_cs_all   = convert_df_to_matrix(df_agg, gene_list_dict=None, stat="mean", exp=False).clip(lower=0)

    infoLine(f"Select candidate marker genes with top contribution score (TopN={TopN})")
    
    # Pass significant_gene_names to the selection function
    marker_gene_dict = select_top_contribution_markers(
        df_agg=df_agg, 
        df_median=df_median_cs_all, 
        df_mean=df_mean_cs_all, 
        significant_gene_names=significant_gene_names,
        top_n=TopN
    )

    # Summary statistics
    total_selected = sum(len(genes) for genes in marker_gene_dict.values())
    infoLine(f"Finish candidate genes selection! Selected {total_selected} genes across {len(marker_gene_dict)} groups")
    
    # Print summary per group
    for group, genes in marker_gene_dict.items():
        infoLine(f"  Group '{group}': {len(genes)} candidate genes")
        if len(genes) > 0:
            infoLine(f"    Top 5: {', '.join(genes[:5])}{'...' if len(genes) > 5 else ''}")

    return marker_gene_dict



#Top marker gene selection
def extractMarkerGenes(file, marker_dict, outDIR, marker_num):
    """
    Extract cell-type-specific marker genes based on silhouette score and expression specificity.

    Parameters:
    - file: Path to expression matrix file (tab-separated, includes 'cell_id', 'group', and gene columns)
    - marker_dict: Dictionary of group -> list of candidate genes (from attribution or contribution scoring)
    - outDIR: Output directory path to write marker gene file
    - marker_num: Number of marker genes to select per group

    Returns:
    - top_genes: Dictionary of group -> list of selected marker genes
    """
    infoLine("Input expression data")
    df_exp_input = pd.read_csv(file, sep='\t')

    def compute_fraction_from_df(df, marker_dict, threshold=1):
        gene_list = set(sum(marker_dict.values(), []))
        
        available_genes = [g for g in gene_list if g in df.columns]
        if not available_genes:
            raise ValueError("No overlap genes between marker_dict and input file")
    
        df_selected = df[["group"] + available_genes]
    
        cell_types = df['group']
        expr_mat = df_selected.iloc[:, 1:]  
    
        result = {}
    
        for ct in cell_types.unique():
            group_expr = expr_mat[cell_types == ct]
            binary_expr = (group_expr > threshold).astype(int)
            frac = binary_expr.sum(axis=0) / group_expr.shape[0]
            result[ct] = frac
    
        frac_df = pd.DataFrame(result)
        return frac_df  # cell_type × gene
    
    # ---------- Silhouette Score ----------
    def compute_silhouette_score(df, marker_dict):
        groups = df['group'].astype(str)
        expr_df = df.drop(columns=['cell_id', 'group'])
        result = {}
        for group, genes in marker_dict.items():
            binary_label = (groups == group).astype(int).values
            gene_scores = {}
            for gene in genes:
                if gene not in expr_df.columns:
                    continue
                try:
                    val = expr_df[gene].values.reshape(-1, 1)
                    score = silhouette_score(val, binary_label)
                    gene_scores[gene] = score
                except Exception:
                    continue
            # result[group] = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
            result[group] = [(gene, gene_scores[gene]) for gene in genes if gene in gene_scores]

        return result

    # ---------- Merge with Expression ----------
    def merge_result(sil_dict, df_fraction, df_median, df_mean, marker_dict):
        records = []
        for group, gene_scores in sil_dict.items():
            for gene, sil_score in gene_scores:
                if gene not in marker_dict.get(group, []):
                    continue
                median_expr = df_median.at[gene, group] if gene in df_median.index and group in df_median.columns else None
                mean_expr = df_mean.at[gene, group] if gene in df_mean.index and group in df_mean.columns else None
                cell_fraction = df_fraction.at[gene, group] if gene in df_fraction.index and group in df_fraction.columns else None
                records.append({
                    "group": group,
                    "gene": gene,
                    "silhouette_score": sil_score,
                    "cell_fraction": cell_fraction,
                    "median_expr": median_expr,
                    "mean_expr": mean_expr
                })
        return pd.DataFrame.from_records(records)

    # ---------- Select Markers ----------
    def select_marker_genes(result_df, df_fraction, df_mean, df_median, top_n=20,
                        initial_fold=2.0, fold_step=0.1, end_fold=1.2, eps=1e-6):

        selected_dict = {}
        records = []  
        
        df_mean.columns = df_mean.columns.astype(str)
        df_median.columns = df_median.columns.astype(str)
        df_fraction.columns = df_fraction.columns.astype(str)
        result_df["group"] = result_df["group"].astype(str)
    
        for group in result_df['group'].unique():
            group = str(group)
            genes = df_mean.index
    
            # target celltype expression
            t_mean = df_mean[group].reindex(genes).fillna(eps)
            t_median = df_median[group].reindex(genes).fillna(eps)
            t_frac = df_fraction[group].reindex(genes).fillna(eps)
    
            # non-target celltype expression
            other_cols = [c for c in df_mean.columns if c != group]
            o_mean_sum = df_mean[other_cols].sum(axis=1).reindex(genes).fillna(eps)
            o_median_sum = df_median[other_cols].sum(axis=1).reindex(genes).fillna(eps)
            o_frac_sum = df_fraction[other_cols].sum(axis=1).reindex(genes).fillna(eps)
    
            o_mean_max = df_mean[other_cols].max(axis=1).reindex(genes).fillna(eps)
            o_median_max = df_median[other_cols].max(axis=1).reindex(genes).fillna(eps)
            o_frac_max = df_fraction[other_cols].max(axis=1).reindex(genes).fillna(eps)
    
            # spec score
            spec_mean = np.log2((t_mean + eps) / (o_mean_sum + eps))
            spec_median = np.log2((t_median + eps) / (o_median_sum + eps))
            spec_frac = np.log2((t_frac + eps) / (o_frac_sum + eps))
    
            # Min-Max 
            def minmax_norm(s): 
                return (s - s.min()) / (s.max() - s.min() + eps)
    
            specificity = minmax_norm(spec_mean) + minmax_norm(spec_median) + minmax_norm(spec_frac)
    
            # marker_score
            result_sub = result_df[result_df['group'] == group].copy()
            result_sub = result_sub.drop_duplicates('gene')
            genes = result_sub['gene'].tolist()
            sil_scores = result_sub.set_index('gene')['silhouette_score']
            marker_score = sil_scores.reindex(genes).fillna(0) * specificity
    
            # selection top_n marker genes
            if initial_fold < end_fold:
                raise ValueError("initial_fold should be >= end_fold")
            folds = np.arange(initial_fold, end_fold - 1e-9, -fold_step)
    
            selected_genes = []
            gene2fold = {}  
    
            for fold in folds:
                cond = (t_mean > fold * o_mean_max) & (t_median > fold * o_median_max) & (t_frac > fold * o_frac_max)
                cond = cond.reindex(marker_score.index, fill_value=False)
    
                fold_genes = marker_score[cond].sort_values(ascending=False).index.tolist()
                fold_genes = [g for g in fold_genes if g not in selected_genes]
    
                for g in fold_genes:
                    if g not in gene2fold:
                        gene2fold[g] = fold
    
                if len(selected_genes) + len(fold_genes) >= top_n:
                    needed = top_n - len(selected_genes)
                    selected_genes += fold_genes[:needed]
                    break
                else:
                    selected_genes += fold_genes
    
            # fallback top_n
            if len(selected_genes) < top_n:
                needed = top_n - len(selected_genes)
                group_genes = result_df[result_df['group'] == group]['gene'].tolist()
                for g in group_genes:
                    if g not in selected_genes:
                        selected_genes.append(g)
                        gene2fold[g] = np.nan 
                        needed -= 1
                        if needed == 0:
                            break
    
            selected_dict[group] = selected_genes
    
            for g in selected_genes:
                records.append({
                    "marker": g,
                    "target_cell_type": group,
                    "fold_level": gene2fold.get(g, np.nan),
                    "marker_score": marker_score.get(g, np.nan)
                })
    
        df_selected = pd.DataFrame(records)
        return selected_dict, df_selected

    # ---------- Export marker list----------
    def export_marker(df_selected, output_path):
        cols = ["marker", "target_cell_type", "fold_level", "marker_score"]
        df_selected.to_csv(output_path, sep='\t', index=False, columns=cols)

    # ---------- Pipeline ----------
    infoLine("Calculate median and mean expression")
    df_median = convert_df_to_matrix(df_exp_input, marker_dict, stat="median", exp=True)
    df_mean = convert_df_to_matrix(df_exp_input, marker_dict, stat="mean", exp=True)

    infoLine("Calculate silhouette score")
    sil_scores = compute_silhouette_score(df_exp_input, marker_dict)

    infoLine("Calculate cell fraction")
    df_fraction = compute_fraction_from_df(df_exp_input, marker_dict)
    
    infoLine("Select top marker genes")
    result_df = merge_result(sil_scores, df_fraction, df_median, df_mean, marker_dict)
    selected_dict, df_selected = select_marker_genes(result_df, df_fraction, df_mean, df_median, marker_num)
    
    infoLine("Output marker genes")
    export_marker(df_selected, outDIR + f'/scMarkerGene_Top{marker_num}_marker.txt')

    infoLine("Finish marker selection!")
    return selected_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version: 1.0 \nDescription: prepare datasets for neural network",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", dest="inDIR",        type=str, required=True,  help="Path of input data")
    parser.add_argument("-o", dest="outDIR",       type=str, required=True,  help="The directory of output data")
    parser.add_argument("-e", dest="exp_data",     type=str, required=True,  help="The directory of expression data")
    parser.add_argument("-p", dest="n_perm",       type=int, required=False, help="Number of permutation test", default=1000)
    parser.add_argument("-n", dest="marker_num",   type=int, required=False, help="Marker genes in each cell", default=20)
    parser.add_argument("-t", dest="TopN",         type=int, required=False, help="Number of first selection genes with contribution score", default=50)
    
    args=parser.parse_args()
    
    inDIR        = args.inDIR 
    outDIR       = args.outDIR
    exp_data     = args.exp_data
    n_perm       = args.n_perm
    TopN         = args.TopN 
    marker_num   = args.marker_num 
 
    #Aggregating all models
    infoLine("Aggregating all models")
    significant_gene_names = aggregateModel(inDIR, outDIR, n_perm)

    #candidate genes selected by cpntribution score
    infoLine("Extract candidate genes")
    candidate_genes = extractPotentialGenes(inDIR, outDIR, TopN, significant_gene_names)
    
    #Create marker genes file
    infoLine("Select marker genes")
    extractMarkerGenes(exp_data, candidate_genes, outDIR, marker_num)
    
    infoLine("Done!")
