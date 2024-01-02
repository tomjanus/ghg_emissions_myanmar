""" Defines functions for dimensionality reduction"""
import os
from typing import Tuple, Optional, Literal, Union
from collections.abc import Iterable, Collection
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE, MDS
from skbio.stats.ordination import pcoa
from sklearn.metrics import euclidean_distances
from umap import UMAP
import skbio
from skbio.stats.ordination import pcoa


PCAModel = sklearn.decomposition._pca.PCA
FAModel = sklearn.decomposition._factor_analysis.FactorAnalysis
ScalingStrategy = Union[Normalizer, RobustScaler, StandardScaler, MinMaxScaler]
PCOAModel = skbio.stats.ordination._ordination_results.OrdinationResults


def calculate_stress(mds, data, verbose: bool = True) -> float:
    """Function for calculating Kruskal's stress value in MuldiDimensional Scaling (MDS)"""
    # Coordinates of points in the plan (n_components=2)
    points = mds.embedding_
    ## Manual calculus of sklearn stress
    DE = euclidean_distances(points)
    stress = 0.5 * np.sum((DE - data.values)**2)
    ## Kruskal's stress (or stress formula 1)
    stress_kruskal = np.sqrt(stress / (0.5 * np.sum(data.values**2)))
    if verbose:
        print("Kruskal's Stress :")
        print("[Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]")
        print(stress_kruskal)
        print("")
    return stress_kruskal
    
    
def run_mds(
        X: np.ndarray | pd.DataFrame, 
        y: np.ndarray | pd.Series | None = None, 
        scaling_strategy: ScalingStrategy | None = None, 
        dissimilarity: str = 'precomputed',
        metric : bool = True,
        n_components: int = 2, random_state: int =42, verbose: bool = True):
    """Runs multidimensional scaling using a pre-computed matrix of distances between points"""
    print(f'Running dimensionality reduction with multidimensional scaling (MDS) on a distance matrix')
    if scaling_strategy:
        pipeline = make_pipeline(
            scaling_strategy(), 
            MDS(
                dissimilarity=dissimilarity, metric = metric, n_components=n_components, 
                random_state=random_state))
    else:
        pipeline = make_pipeline( 
            MDS(
                dissimilarity=dissimilarity, metric = metric, n_components=n_components, 
                random_state=random_state))
    X_mds = pipeline.fit_transform(X)
    if verbose and dissimilarity == 'precomputed':
        calculate_stress(pipeline.named_steps['mds'], pd.DataFrame(X), verbose = True)
    return X_mds, y, pipeline.named_steps['mds']


def run_pcoa(
        X: np.ndarray | pd.DataFrame, 
        y: np.ndarray | pd.Series | None = None,
        scaling_strategy: ScalingStrategy | None = None,
        n_components: int | None = None):
    """Runs principal coordinate analysis using distance matrix between points as an input"""
    print(f'Running dimensionality reduction with principal coordinate analysis (PCoA) on a distance matrix')
    # Create a list of PC component names based on the number of components
    if scaling_strategy:
        X = scaling_strategy().fit_transform(X)
    if n_components:
        pcoa_model = pcoa(X, number_of_dimensions=n_components)
    else:
        pcoa_model = pcoa(X)
    # Retrieve information from the 'trained' PCoA model
    pca_ranks: pd.Series = pcoa_model.proportion_explained
    X_pcoa: pd.DataFrame = pcoa_model.samples
    X_pcoa.columns = X_pcoa.columns.str.replace('PC', 'PCo')
    
    return X_pcoa, y, pcoa_model, pca_ranks
            

def run_pca(
        X, y, 
        scaling_strategy: ScalingStrategy | None = None,
        n_components: int | None = None,
        random_state: int | None = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    print(f'Running dimensionality reduction with principal component analysis (PCA)')
    if scaling_strategy:
        pipeline = make_pipeline(
            scaling_strategy(), 
            PCA(n_components=n_components, random_state=random_state))
    else:
        pipeline = make_pipeline( 
            PCA(n_components=n_components, random_state=random_state))
    X_pca = pipeline.fit_transform(X)
    return X_pca, y, pipeline.named_steps['pca']


def run_fa(
        X, y, 
        scaling_strategy: ScalingStrategy | None = None,
        n_components: int | None = None,
        rotation: str | None = "quartimax") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    print(f'Running dimensionality reduction with factor analysis analysis (FA)')
    if scaling_strategy:
        pipeline = make_pipeline(
            scaling_strategy(), 
            FactorAnalysis(n_components=n_components, rotation=rotation))
    else:
        pipeline = make_pipeline(
            FactorAnalysis(n_components=n_components, rotation=rotation))        
    X_fa = pipeline.fit_transform(X)
    return X_fa, y, pipeline.named_steps['factoranalysis']


def run_tsne(
        X, y,
        scaling_strategy: ScalingStrategy | None = None,
        n_components: int = 3,
        random_state: int | None = 42,
        perplexity: int = 50):
    """ """
    print(f'Running dimensionality reduction with t-SNE')
    if scaling_strategy:
        pipeline = make_pipeline(
            scaling_strategy(), 
            TSNE(
                n_components=n_components, 
                random_state=random_state, 
                perplexity=perplexity))
    else:
        pipeline = make_pipeline( 
            TSNE(
                n_components=n_components, 
                random_state=random_state, 
                perplexity=perplexity))        
    X_tsne = pipeline.fit_transform(X)
    print(f'TSNE divergence: {pipeline.named_steps["tsne"].kl_divergence_}')
    return X_tsne, y, pipeline.named_steps['tsne']
    
    
def run_umap(
        X, y, scaling_strategy: ScalingStrategy | None = None,
        n_neighbors: int = 15, min_dist: float = 0.1, 
        n_components: int = 2,
        metric: Literal['correlation', 'euclidean'] = 'euclidean',
        random_state: int | None = 42):
    """ """
    print(f'Running dimensionality reduction with UMAP')
    if scaling_strategy:
        pipeline = make_pipeline(
            scaling_strategy(), 
            UMAP(
                n_neighbors=n_neighbors, 
                n_components = n_components,
                random_state=random_state,
                metric = metric, 
                min_dist = min_dist))
    else:
        pipeline = make_pipeline( 
            TUMAP(
                n_neighbors=n_neighbors, 
                n_components = n_components,
                random_state=random_state,
                metric = metric, 
                min_dist = min_dist))        
    X_umap = pipeline.fit_transform(X)
    return X_umap, y, pipeline.named_steps['umap']
    
    
def explained_cumulative_var_plot(
        ordination_model: PCAModel | PCOAModel, 
        title: str | None = None,
        xlabel: str = "Number of components",
        ax: matplotlib.axes._axes.Axes | None = None,
        num_components: int | None = None,
        figsize: Tuple[int, int] = (6,4)) -> None:
    """Plots explained and cumulative explained variances for a PCA model"""
    # Explained variances and cumulative explained variances
    if isinstance(ordination_model, PCAModel):
        explained_variances = ordination_model.explained_variance_ratio_
    elif isinstance(ordination_model, PCOAModel):
        explained_variances = ordination_model.proportion_explained
    else:
        raise ValueError("Unsupported Model Type")
    cumulative_variances = np.cumsum(explained_variances)
    if not title:
        title = 'Explained and Cumulative Explained Variances'
    # Reduce number of items if num_components supplied as argument
    if num_components:
        explained_variances = explained_variances[:num_components]
        cumulative_variances = cumulative_variances[:num_components]
    # Plot pca
    if ax == None:
        plt.figure(figsize=figsize)
        ax = plt
        plt.xlabel(xlabel)
        plt.ylabel('Explained variance, (-)')
        plt.title(title)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Explained variance, (-)')
        ax.set_title(title)
    ax.plot(
        range(1, len(explained_variances) + 1), explained_variances, marker='o',
        linestyle='-', label='Explained Variance')
    ax.plot(
        range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o',
        linestyle='--', label='Cumulative Explained Variance')

    ax.legend()
    ax.grid(True)
    if ax == plt:
        plt.show()
        
        
def features_to_vars(
        dim_red_model: PCAModel | FAModel,
        column_names: Collection,
        output_file: str | None = None) -> pd.DataFrame:
    """Create a pandas dataframe matching dimensionality reduction model directions, e.g.
    principal components to variable names in the data used for dimensionality reduction.
    Uses either PCA or FA algorithm outputs.
    Optionally saves the dataframe to an excel file
    """
    
    def is_excel_file(file_path: str):
        _, file_extension = os.path.splitext(file_path)
        return file_extension.lower() in ['.xls', '.xlsx']

    if isinstance(dim_red_model, PCAModel):
        index_name: str = 'PC'
    elif isinstance(dim_red_model, FAModel):
        index_name = 'FA'
    else:
        raise ValueError(f"Model {dim_red_model} has unsupported type {type(dim_red_model)}.")
    
    feat_to_vars = pd.DataFrame(
        dim_red_model.components_,
        columns=column_names,
        index = [f'{index_name}-{index}' for index in range(0, dim_red_model.components_.shape[0])])
    
    if output_file:
        if is_excel_file(output_file):
            pca_to_feature.to_excel(output_file)
        else:
            print('Provided file has incompatible extension. Only .xls and .xlsx supported.')
            
    return feat_to_vars
    
    
def plot_component_feature_map(
        model_to_feat: pd.DataFrame,
        num_dims: int | None = 20,
        num_feats: int | None = None, 
        figsize: Tuple[float, float] = (16,10), 
        ticks: Tuple[float, ...] = (-1, -0.5, 0, 0.5, 1),
        xtick_rotation: float = 90,
        ytick_rotation: float = 0) -> None:
    """Plots a clustermap showing relationships between dimensions in dimensionally reduced
    space and features in the original data"""
    if not num_dims:
        num_dims = model_to_feat.shape[0]
    if not num_feats:
        num_feats = model_to_feat.shape[1]
    fig = plt.figure()
    # Colorbar options
    kws = dict(
        cbar_kws=dict(
            ticks=ticks, orientation='vertical', label="Feature Rank",
            shrink=0.8, extend = 'both', drawedges = False, fraction=0.02))
    # Create the clustermap
    clustermap = sns.clustermap(
        model_to_feat.iloc[0:num_dims,0:num_feats], annot=False, cmap='RdYlGn_r', 
        linewidths=0.5, row_cluster=False,
        metric='euclidean', method='complete', figsize=figsize, 
        cbar_pos=(0.08, 0.78, 0.05, 0.20),
        xticklabels=list(model_to_feat.columns[0:num_feats]),
        **kws)
        
    plt.setp(
        clustermap.ax_heatmap.yaxis.get_majorticklabels(), 
        rotation=ytick_rotation)
    plt.setp(
        clustermap.ax_heatmap.xaxis.get_majorticklabels(), 
        rotation=xtick_rotation)

    # Apply tight layout so that no content goes over the edge, etc.
    fig.tight_layout()
