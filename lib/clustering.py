"""Runners of clustering algorithms that are implemented in findings groups of reservoirs
that have similar properties in the way inputs affect the predictions of their CO2 and CH4
emissions"""
from typing import Tuple, List, Literal
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import get_cmap
from matplotlib.cm import Dark2, Set3
import sklearn
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, KMeans, AgglomerativeClustering, cluster_optics_dbscan
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
import skbio


PCAModel = sklearn.decomposition._pca.PCA
PCOAModel = skbio.stats.ordination._ordination_results.OrdinationResults


def run_agglomerative(
        X: np.ndarray, n_dim: int | None = None, n_clusters: int | None = None,
        metric: str | None = None, distance_threshold: float | None = None, 
        col_name: str = 'cluster') -> pd.DataFrame:
    """Runs clustering with agglomerative clustering method in scikit-learn"""
    if n_dim:
        X = X[:,:n_dim]
    if metric == "precomputed":
        linkage = "complete"
    else:
        linkage = "ward"
    agg_clust = AgglomerativeClustering(
        n_clusters = n_clusters, metric = metric,
        linkage = linkage,
        distance_threshold = distance_threshold).fit(X)
    labels = agg_clust.labels_
    s_score = silhouette_score(X, labels)
    print("------------Agglomerative clustering----------------")
    print(f"Silhouette coefficient Agglomerative Clustering: {s_score:.3f}")
    print("\n")
    return pd.DataFrame(labels, columns=[col_name])


def run_kmeans(
        X: np.ndarray, n_dim: int | None = None, n_clusters: int = 5, 
        col_name: str = 'cluster') -> pd.DataFrame:
    """Runs K mean clustering on data X and returns a dataframe of labels"""
    kmeans = KMeans(n_clusters=n_clusters, n_init = 'auto')
    if n_dim:
        X = X[:,:n_dim]
    labels = kmeans.fit_predict(X)
    s_score = silhouette_score(X, labels)
    print("------------K-Means clustering----------------")
    print(f"Silhouette coefficient K-Means: {s_score:.3f}")
    print("\n")
    return pd.DataFrame(labels, columns=[col_name])
    

def run_kmedoids(
        X: np.ndarray, n_clusters: int = 5, n_dim: int | None = None, 
        metric: str = 'euclidean', col_name: str = 'cluster', 
        method: Literal['alternate', 'pam'] = 'alternate',
        random_state: int | None = None) -> pd.DataFrame:
    """Runs K medoids clustering algorithm"""
    kmedoids = KMedoids(
        n_clusters=n_clusters, random_state=random_state,
        metric = metric, method = method)
    if n_dim:
        X = X[:,:n_dim]
    kmedoids = kmedoids.fit(X)
    labels = kmedoids.labels_
    centers = kmedoids.cluster_centers_
    s_score = silhouette_score(X, labels)
    print("------------K-Medoids clustering----------------")
    print(f"Silhouette coefficient K-Medoids: {s_score:.3f}")
    print("\n")
    return pd.DataFrame(labels, columns=[col_name])

    
def run_optics(
        X: np.ndarray, min_samples:  int = 5, 
        xi: float = 0.05, min_cluster_size: float = 0.05,
        metric: str = 'minkowski',
        cluster_method: str = 'xi',
        n_dim: int | None = None, col_name: str = 'cluster') -> pd.DataFrame:
    """ """
    print("-------------OPTICS clustering---------------")
    optics_model = OPTICS(
        min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size,
        cluster_method = cluster_method,
        metric = metric)
    if n_dim:
        X = X[:,:n_dim]
    optics_model.fit(X)
    optics_labels = optics_model.labels_[optics_model.ordering_]
    optics_labels_df = pd.DataFrame(optics_labels, columns=[col_name])
    s_score = silhouette_score(X, optics_labels)
    print(f"Silhouette coefficient OPTICS: {s_score:.3f}")
    print("\n")
    return optics_labels_df 

    
def run_dbscan(
        X: np.ndarray, n_dim: int | None = None, eps: float = 0.0375, 
        min_samples: int = 50, col_name: str = 'cluster') -> pd.DataFrame:
    """ """
    if n_dim:
        X = X[:,:n_dim]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    dbscan_labels = db.labels_
    dbscan_labels_df = pd.DataFrame(dbscan_labels, columns=[col_name])
    s_score = silhouette_score(X, dbscan_labels)
    print("--------------DBSCAN clustering--------------")
    print(f"Silhouette coefficient DBSCAN: {s_score:.3f}")
    # Run statistics on noise points
    n_noise = list(dbscan_labels).count(-1)
    print("Number of noise points in DBSCAN:", n_noise)
    print("\n")
    return dbscan_labels_df


def run_hdbscan(
        X: np.ndarray, n_dim: int | None = None, 
        min_samples: int = 50,
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 0.0001,
        min_cluster_size: int = 5,
        metric: str = 'euclidean',
        col_name: str = 'cluster') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ """
    hdb = HDBSCAN(
        min_samples = min_samples, metric = metric, min_cluster_size = min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon, alpha = alpha)
    if n_dim:
        X = X[:,:n_dim]
    hdb.fit(X)
    hdbscan_labels = hdb.labels_
    hdbscan_labels_df = pd.DataFrame(hdbscan_labels, columns=[col_name])
    hdbscan_probabilities_df = pd.DataFrame(hdb.probabilities_, columns=['probability'])
    s_score = silhouette_score(X, hdbscan_labels)
    print("-------------HDBSCAN clustering---------------")
    print(f"Silhouette coefficient HDBSCAN: {s_score:.3f}")
    # Run statistics on noise points
    n_noise = list(hdbscan_labels).count(-1)
    print("Number of noise points in HDBSCAN:", n_noise)
    print("\n")
    return hdbscan_labels_df, hdbscan_probabilities_df

    
def run_gmm(
        X: np.ndarray, n_clusters: int = 5, n_dim: int | None = None,
        init_params: str = 'kmeans',
        col_name: str = 'cluster') -> pd.DataFrame:
    """Run Gaussian Mixture Model clustering"""
    gmm = GaussianMixture(
        n_components = n_clusters, random_state = 666, covariance_type='full',
        init_params = init_params)
    if n_dim:
        X = X[:, :n_dim]
    gmm.fit(X)
    cluster_centres = gmm.means_
    gmm_labels = gmm.predict(X)
    gmm_probabilities = gmm.predict_proba(X)
    #gmm_probabilities_df = pd.DataFrame(gmm_probabilities, columns=['probability'])
    gmm_labels_df = pd.DataFrame(gmm_labels, columns=[col_name])
    s_score = silhouette_score(X, gmm_labels)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    print("---------------GMM clustering-----------------")
    print(f"Silhouette coefficient Gaussian Mixture Model: {s_score:.3f}")
    print(f"Akaike Information Criterion: {-aic:.3f}")
    print(f"Bayes Information Criterion: {-bic:.3f}")
    print("\n")
    return gmm_labels_df
    
    
def run_bgmm(
    X: np.ndarray, n_clusters: int = 5, n_dim: int | None = None,
        init_params: str = 'kmeans',
        col_name: str = 'cluster') -> pd.DataFrame:
    """Run Bayesian Gaussian Mixture Model clustering"""
    bgmm = BayesianGaussianMixture(
        n_components = n_clusters, random_state = 666, covariance_type = 'full',
        init_params = init_params)
    if n_dim:
        X = X[:, :n_dim]
    bgmm.fit(X)
    cluster_centres = bgmm.means_
    bgmm_labels = bgmm.predict(X)
    bgmm_labels_df = pd.DataFrame(bgmm_labels, columns=[col_name])
    s_score = silhouette_score(X, bgmm_labels)
    print("---------------BGMM clustering----------------")
    print(f"Silhouette coefficient Gaussian Mixture Model: {s_score:.3f}")
    print("\n")
    return bgmm_labels_df


def plot_kneighbours_dist_graph(
        X: np.ndarray, 
        n_dim: int | None = None, 
        scaler: None | Normalizer | StandardScaler | RobustScaler = None,
        cutoff_line_value: float = 0.7,
        title: str | None = None,
        ax = None) -> None:
    """ """
    if not title:
        title = 'K-Neighbors distance graph'
    if n_dim:
        X = X[:,:n_dim]
    if scaler:
        X = scaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=4) # minimum points -1
    nbrs = nn.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    # Choose only the smallest distances
    distances = distances[:,1]
    if not ax:
        plt.figure(figsize=(6,3))
        plt.title(title)
        plt.xlabel('Data points')
        plt.ylabel('Distance')
        ax = plt
    else:
        ax.set_title(title)
        ax.set_xlabel('Data points')
        ax.set_ylabel('Distance')

    ax.plot(distances)
    ax.axhline(y=cutoff_line_value, color='r', linestyle='--', alpha=0.4) # elbow line
    if ax == plt:
        plt.show()


def visualise_clusters_2D(
        data: np.ndarray,
        labels: pd.DataFrame,
        s_multiplier: float,
        title: str | None = None,
        probabilities: pd.DataFrame | None = None,
        ax = None,
        default_alpha: float = 0.5):
    """ """
    # Set alpha based on the level of probability if probabilities given
    # Otherwise, use a preset alpha factor
    if probabilities is not None:
        alpha_factor = probabilities['probability']
    else:
        alpha_factor = default_alpha

    if ax is None:
        show_figure: bool = True
        ax = plt.gca()
    else:
        show_figure = False
        
    if data.shape[1] < 3:
        point_size = s_multiplier
    else:
        point_size = s_multiplier*np.abs(data[:,2])
    
    # Plot in 2D with first and second component on x and y axis, respectively
    # Third component is given as point size
    ax.scatter(
        data[:,0], data[:,1], c=labels['cluster'], 
        s=point_size, edgecolor = 'k', 
        alpha=alpha_factor,
        linewidth=0.3)
    
    # If probabilities are given, some of the points might have zero alpha and thus,
    # be invisible. Therefore, overlay the points with point outlines with transparent background
    if probabilities is not None:
        ax.scatter(
            data[:,0], data[:,1], facecolor="none", 
            s=point_size, 
            edgecolor = 'k', linewidth=0.3)
        
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_title(title, fontsize = 12)
    
    plt.tight_layout()
    
    if show_figure:
        plt.show()
        
    return plt


def visualise_pca_2D(
        data: np.ndarray,
        labels: pd.DataFrame,
        var_names: List[str],
        pca_model: PCAModel | PCOAModel,
        s_multiplier: float,
        cmap = get_cmap('Set3', 10),
        title: str | None = None,
        probabilities: pd.DataFrame | None = None,
        num_components: int = 5, # Number of principal components shown on the plot
        ax = None, 
        default_alpha: float = 0.5,
        plot_ellipses: bool = True,
        arrow_width: float = 0.0125,
        legend_location: str = 'lower left',
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        tick_fontsize: int = 10): 
    """ """
    def arrow_lengths(arrows) -> np.array:
        return (np.sum(arrows**2, axis=1)).T
    
    pos_translator = {
        0 : (1.2, 1.1),
        1 : (1.0, 0.9),
        2 : (2.0, 1.3),
        3 : (1.8, 1.5)
    }
    
    # Process labels
    cluster_ids = sorted(labels['cluster'].unique())
    n_unique = len(cluster_ids)
    
    plot_arrows: bool = False
    n_dim = 2 # 2D plot
    scores = data[:, :n_dim]
    if isinstance(pca_model, PCAModel):
        loadings = pca_model.components_[:n_dim].T
        pvars = pca_model.explained_variance_ratio_[:n_dim] * 100
        plot_arrows = True
        component_name = "PC"
    elif isinstance(pca_model, PCOAModel):
        loadings = pca_model.samples.iloc[:,:n_dim].to_numpy()
        pvars = pca_model.proportion_explained[:n_dim] * 100
        component_name = "PCo"
    else:
        raise ValueError("Wrong model type")
    
    # proportions of variance explained by axes
    arrows = loadings * np.abs(scores).max(axis=0)
    indices_sorted = np.argsort(arrow_lengths(arrows))[::-1]
    if num_components > len(loadings):
        num_components = len(loadings)
        
    if probabilities is not None:
        alpha_factor = probabilities['probability']
    else:
        alpha_factor = default_alpha
        
    if ax is None:
        show_figure: bool = True
        ax = plt.gca()
    else:
        show_figure = False
        
    if plot_arrows:
        # empirical formula to determine arrow width
        width = -arrow_width * np.min([np.subtract(*plt.xlim()), np.subtract(*plt.ylim())])
        # features as arrows
        texts = []
        for i in range(0,num_components):  #, arrow in enumerate(arrows):
            index = indices_sorted[i]
            arrow = 0.8 * arrows[index,:]
            x_arr, y_arr = arrow[0] * 1.10, arrow[1] * 1.10
            x_arr, y_arr = x_arr * pos_translator[i][0], y_arr * pos_translator[i][1]
            ax.arrow(0, 0, *arrow, color='black', alpha=1, width=width, ec='none',zorder=5,
                      length_includes_head=True)
            ax.text(x_arr, y_arr, var_names[index], snap=False, fontvariant='small-caps',
                     ha='center', va='center', fontsize=11, style='italic', color = 'black',
                     bbox={
                         'facecolor': 'none', 'alpha': 0.3, 'pad': 2, 'linewidth': 0.0, 
                         'capstyle': 'round'})
            
        #adjust_text(
        #    texts, ax = ax, 
        #    arrowprops=dict(arrowstyle="-", color='k', lw=0.3, alpha=0.3), expand_objects =(1.8, 1.8),
        #    expand_text=(1.8, 1.8))     
        
    #ax.scatter(
    #    data[:,0], data[:,1], c=labels['cluster'], s=s_multiplier*np.abs(data[:,2]), 
    #    edgecolor='k', linewidth=0.5, alpha=alpha_factor)

    
    for ix, label in enumerate(cluster_ids):
        #print(cmap)
        ax.scatter(data[labels['cluster'] == label, 0], data[labels['cluster'] == label, 1], 
                   linewidth=0.5, s=s_multiplier*np.abs(data[labels['cluster'] == label, 2]),
                   color = cmap.colors[ix], alpha=0.7,
                   edgecolor = 'k') # , facecolor="none", edgecolor = 'k'
        if plot_ellipses : #or not plot_ellipses: # In any case
            confidence_ellipse(data[labels['cluster'] == label, 0], data[labels['cluster'] == label, 1], 
                               ax=ax, label = f'Cluster {label}',
                               edgecolor='k', n_std = 3, linewidth=0.5, zorder=0,
                               facecolor=cmap.colors[ix], alpha=alpha_factor)
    ax.axvline(c='grey', lw=1, linestyle='--')
    ax.axhline(c='grey', lw=1, linestyle='--')
    
    #if probabilities is not None:
    #    ax.scatter(
    #        data[:,0], data[:,1], facecolor="none", s=s_multiplier*np.abs(data[:,2]), 
    #        edgecolor = 'k', linewidth=0.5)

    ax.set_title(title, fontsize = title_fontsize)
    #plt.xlim([-25, -10])
    #plt.ylim([-10, 40])
    # axis labels
    ax.set_xlabel(f'{component_name}$_{1}$ ({pvars[0]:.2f}% explained variance)')
    ax.set_ylabel(f'{component_name}$_{2}$ ({pvars[1]:.2f}% explained variance)')    
    xlabel_format = '{:,.2f}'
    ylabel_format = '{:,.2f}'
    
    yticks_loc = ax.get_yticks().tolist()
    ytick_labels = ax.get_yticklabels()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels([ylabel_format.format(x) for x in yticks_loc], fontsize = tick_fontsize)
    xticks_loc = ax.get_xticks().tolist()
    xtick_labels = ax.get_xticklabels()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels([xlabel_format.format(x) for x in xticks_loc], fontsize = tick_fontsize)
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    
    l1 = ax.legend(
        loc=legend_location, borderpad = 0.6, edgecolor = 'black' , facecolor = 'none', 
        shadow=False, labelspacing = 0.6, framealpha = 0.9,
        prop={'size': 11, 'style': 'italic'} )
    l1.get_frame().set_linewidth(0.0)
    plt.tight_layout()
    if show_figure:
        plt.show()
        
    return plt
        
        
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', xy_scale_factor: int = 1.20, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * xy_scale_factor,
                      height=ell_radius_y * xy_scale_factor,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = mpl.transforms.Affine2D() \
        .rotate_deg(22.5) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
    
# CODE FOR ADDING ANNOTATIONS, E.G. RESERVOIR NAMES TO PLOTS
#for label, x, y in zip(col1.index, col1, col2):
#    texts+=[ax.text(x, y, label, color=groupColors.get(langnameGroup[label],'k'), fontsize=8)] # for adjustText
#from adjustText import adjust_text
#texts = []
#for k, v in df_pcoa_ranks.iterrows():
#    texts.append(plt.text(v['PC1'], v['PC2'], s=v['reservoir name'], alpha = 0.7, fontsize=8))
#adjust_text(texts, ax = ax, arrowprops=dict(arrowstyle="-", color='k', lw=0.3, alpha=0.3), expand_objects =(1.2, 1.2),
            #force_text = (0.25, 0.25),
#            expand_text=(1.2, 1.2))
#for k, v in df_pcoa_ranks.iterrows():
#    ax.annotate(v['reservoir name'], v[['PC1', 'PC2']], alpha = 0.4, fontsize=10)
