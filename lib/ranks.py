""" """
from typing import List, Sequence, Tuple, Dict, Callable
from dataclasses import dataclass, field
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FeatureCorrelationMap:
    """ Description of correlations (0-1) between features (columns) of a numpy.ndarray
    of feature ranks
    Attributes:
        cmap: dictionary between a tuple of column indices (int, int) and correlation value (float)
    """
    cmap: Dict[Tuple[int, int], float]
        
    def get(self, item: Tuple[int, int]) -> float | None:
        """ """
        val = self.cmap.get(item, None)
        if val is None:
            val = self.cmap.get(item[::-1], None)
        return val
        
        
def exp_scaling_fun(
        ix: int, exp_coeff: float = -0.15, min_ix: int | None = 0, max_ix: int = 40, 
        mean_scaler: bool = True) -> float:
    """Exponential scaling function used for scaling rank distances based on their position in the vector
    of ranks sorted by the overall feature importances in the model in the descending order (most
    important features first)"""
    def _fun(k: float, ix: int | np.ndarray) -> float:
        return np.exp(k * ix)
    if mean_scaler:
        scaling_factor = np.mean(_fun(exp_coeff, np.array(range(min_ix, max_ix))))
    else:
        scaling_factor = 1
    return _fun(exp_coeff, ix) / scaling_factor


def rank_similarity_metric(row: pd.Series, reference: pd.Series, weights: pd.Series) -> float:
    """ """
    row_args = [row, reference, weights]
    first_arg = row_args.pop(0)
    for item in row_args:
        try:
            assert len(item) == len(first_arg)
        except AssertionError as err:
            raise AssertionError("pd.Series arguments must have the same lenghts") from err
    N = len(weights)
    metric = sum(abs(row.values-reference.values)*weights.values)/N
    return metric


def create_rank_reference(num_items: int) -> pd.Series:
    """ """
    return pd.Series(range(1,num_items + 1))


def create_mag_order_weights(num_items: int) -> pd.Series:
    """ """
    middle_index = num_items // 2
    end_index = middle_index + num_items % 2
    #if remainder := len(arr) % 2 == 0:
    #    end_index = len(arr)//2
    #else:
    #    end_index = len(arr)//2 + 1
    weights_down = [10 ** i for i in range(middle_index, 0, -1)]
    weights_middle = [1]
    weights_up = [10 ** (-i) for i in range(1, end_index)]

    return pd.Series(weights_down + weights_middle + weights_up)
    

class RankError(Exception):
    pass


def get_ranks(
        feature_data: pd.DataFrame, cols_to_drop: List[str] | None = None, 
        sort_by_means: bool = False, column_order: List[str] | None = None,
        sort_rows: bool = False) -> pd.DataFrame:
    """Get a dataframe of shap values or other feature importances and ranks them
    based on their absolute values with highest absolute values being the most important (rank 1)
    and the lowest values being the least important (rank n, where n is the number of features)"""
    feature_data = feature_data.drop(columns = cols_to_drop)
    ranks = feature_data.copy().abs()
    num_cols = ranks.shape[1]
    columns_r_sorted = False
    
    if sort_by_means:
        column_means = ranks.mean()
        # Sort columns based on mean values
        sorted_columns = column_means.sort_values(ascending=False).index
        columns_r_sorted = True
    else:
        if column_order:
            sorted_columns = column_order
            columns_r_sorted = True
        else:
            sorted_columns = feature_data.columns
    ranks = ranks[sorted_columns]
    #ranks = ranks.sort_values(by=list(sorted_columns[:num_cols]))
        
    # Find ranks of parameters
    for index, row in ranks.iterrows():
        abs_row = row.abs()
        # Rank the features in ascending order and assign ranks to the corresponding cells
        rank_row = abs_row.rank(ascending=False, method='min')
        ranks.loc[index, :] = rank_row
        
    #return ranks.astype(int)

    # Sort and group the rows with ranks
    if sort_rows and not columns_r_sorted:
        raise RankError("Row sorting only possible with sorted columns")
    if sort_rows:
        rank_reference = create_rank_reference(num_cols)
        weights = create_mag_order_weights(num_cols)
        #ranks['Similarity'] = 0
        for ix, row in ranks.iterrows():
            similarity = rank_similarity_metric(row, rank_reference, weights)
            ranks.loc[ix,'Similarity']  = similarity 
        ranks = ranks.sort_values(by='Similarity')
        ranks.drop(columns='Similarity', inplace=True)
        
    # Return ranks converted to integers
    return ranks.astype(int)
    
    
def plot_rank_heatmap(
        rank_data: pd.DataFrame, yticklabels: Sequence['str'], 
        fig_size: Tuple[float, float] = (15,25),
        xlabel_fontsize: int = 10,
        ylabel_fontsize: int = 6,
        linewidths: float = 0.5,
        linecolor: str = 'white',
        cbar_shrink: float = 0.6,
        cbar_ticks: List[int] = None,
        cbar_fraction: float = 0.1) -> None:
    """ """
    fig, ax = plt.subplots(figsize = fig_size)
    if not cbar_ticks:
        cbar_ticks = [1, 10, 20, 30, 40]   
        
    #cmap = sns.color_palette("deep", rank_data.shape[1])
    #summer_r, rainbow
    cmap = sns.color_palette("RdYlGn_r", rank_data.shape[1])

    #rank_data = rank_data.astype('category')
    sorted_columns = rank_data.columns
    heatmap = sns.heatmap(
        rank_data, annot=False, fmt='.2f', linewidths=linewidths, ax=ax, cbar=False,
        cmap=cmap[::-1],
        linewidth=linewidths, linecolor=linecolor, 
        yticklabels=yticklabels,
        xticklabels=sorted_columns)
    # Customize colorbar properties
    cbar_kws = {
        'label': 'Feature Rank',
        'orientation': 'vertical',  # 'horizontal' is the default
        'shrink': cbar_shrink,              # Adjust the size of the colorbar
        'extend': 'min',           # 'neither', 'both', 'min', 'max'
        'ticks': cbar_ticks,
        'fraction': cbar_fraction,
        'drawedges': True,
        'format': '%d'            # Tick label format
    }

    # Apply colorbar customization
    cbar = heatmap.figure.colorbar(heatmap.collections[0], **cbar_kws)
    
    cbar.ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=xlabel_fontsize)
    ax.tick_params(axis='y', labelsize=ylabel_fontsize)


## Find a distance matrix from the ranks dataframe
def rank_distance_matrix(
        rank_df: pd.DataFrame, 
        rank_importances: pd.DataFrame,
        scaling_fun: Callable[[int], float] = partial(
            exp_scaling_fun, exp_coeff = -0.15, min_ix = 0, max_ix = 40, mean_scaler = True),
        corr: FeatureCorrelationMap | None = None) -> pd.DataFrame:
    """Can take quite a long time to compute. Potential improvements - only calculate half of 
    he matrix as it's symmetrical"""
    feat_names = rank_df.columns
    res_indices = rank_df.index
    # Convert data to numpy
    ranks_np = rank_df.to_numpy()
    rank_importances = rank_importances.to_numpy()
    # Create an empy distcance matrix
    no_items = ranks_np.shape[0]
    dist_matrix = np.zeros([no_items, no_items])
    # Calculate distances for each feature pair
    for i in range(0, no_items):
        for j in range(0, no_items):
            row_1 = ranks_np[i, :]
            row_2 = ranks_np[j, :]
            dist_matrix[i,j] = rank_distance(
                row_1, row_2, rank_importances, scaling_fun = scaling_fun, corr=corr)
    # Normalize to zero - 1 scale
    dist_matrix = dist_matrix / dist_matrix.max()
    dist_matrix_df = pd.DataFrame(dist_matrix, index=rank_df.index, columns=rank_df.index)
    return dist_matrix_df
    

def rank_distance(
        arr1: np.array, 
        arr2: np.array, 
        rank_importances: np.array,
        scaling_fun: Callable[[int], float] = partial(
            exp_scaling_fun, exp_coeff = -0.15, min_ix = 0, max_ix = 40, mean_scaler = True),
        corr: FeatureCorrelationMap | None = None) -> float:
    """Custom function for calculating distances between ranks in a ranks matrix"""

    # Ensure the arrays are of equal length
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be of equal length")
    # Discover indices of items that are equal and that are unequal between two vectors
    nomatch_ixs = np.where(arr1 != arr2)[0]

    imp_dist = 0.0
    for index in nomatch_ixs:
        el_1, el_2 = arr1[index], arr2[index] # Two elements that don't match in both vectors 
                                              # (1-based, i.e. 1,2,3...)
        el_dist = np.abs(rank_importances[el_1-1] - rank_importances[el_2-1]).item()
        
        if corr is not None:
            corr_val = corr.get((el_1-1, el_2-1))
            if corr_val is None:
                corr_val = 0.0
        else:
            corr_val = 0.0

        f_scaling = scaling_fun(index)
        imp_dist += el_dist * (1-corr_val) * f_scaling # Reduce difference importance if both elements are correlated
                
    return imp_dist / 2.0

# reference = pd.Series([1,2,3,4,5, 6])
# row = pd.Series([2,1,3,4,5, 6])
# weights = pd.Series([1E+3, 1E+2, 10, 1, 0.1, 0.01])
