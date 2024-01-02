""" """
from typing import Literal, Tuple
import joblib
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import LocalOutlierFactor


def hexbin(x, y, color, **kwargs) -> None:
    """ Create a hexbin plot of two arrays: x and y
    x and y need to be of the same length"""
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

    
def save_model(model, file: str = 'model.pkl', compress: int = 0):
    """Function for saving ML models to local disk space"""
    joblib.dump(model, file, compress)

    
def load_model(file: str) -> None:
    """Function for loading ML models from local disk space"""
    return joblib.load(file)
    
    
def model_check(model, X_train, X_test, y_train, y_test) -> None:
    """ """
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_train =  r2_score(y_train, pred_train)
    r2_test =  r2_score(y_test, pred_test)
    mae_train = mean_absolute_error(y_train, pred_train)/np.mean(y_train) * 100
    mae_test = mean_absolute_error(y_test, pred_test)/np.mean(y_train) * 100
    #mean_absolute_error, mean_squared_error, r2_score
    print(f"MSE_train: {mse_train:.3f}")
    print(f"MSE test: {mse_test:.3f}")
    print(f"MAE_train: {mae_train:.3f} %")
    print(f"MAE test: {mae_test:.3f} %")
    print(f"R2 score train: {r2_train:.3f}")
    print(f"R2 score test: {r2_test:.3f}")


def plot_scores(fit_model, df: pd.DataFrame, n_largest: int = 10, title: str = "Feature scores", ax = None,
        tick_fontsize: int = 8, title_fontsize: int = 10, label_fontsize: int = 8) -> None:
    """ Plots model scores using either f regression or mutual info statistics """
    scores = pd.DataFrame(fit_model.scores_)
    feature_names = pd.DataFrame(df.columns)
    feature_scores = pd.concat([feature_names, scores], axis=1)
    feature_scores.columns = ["Feature", "Score"]
    sorted_feature_scores = feature_scores.sort_values(by="Score", ascending=False)
    if not ax:
        plt = sorted_feature_scores.nlargest(n=n_largest, columns="Score").plot(kind="bar", x="Feature", y="Score", title=title)
    else:
        plt = sorted_feature_scores.nlargest(
            n=n_largest, columns="Score").plot(kind="bar", x="Feature", y="Score", title=title, ax=ax)
    plt.xaxis.set_tick_params(labelsize=tick_fontsize)
    plt.yaxis.set_tick_params(labelsize=tick_fontsize)
    plt.set_title(title, fontsize=title_fontsize)
    plt.set_xlabel('', fontsize=label_fontsize)
    
    
def calculate_gini_feature_importances(X_data, model, max_vars: int|None = None):
    """ """
    feature_importance = model.feature_importances_
    num_features = len(feature_importance)
    if not max_vars or max_vars >= num_features:
        max_vars = num_features
    sorted_idx = np.argsort(feature_importance)[::-1][:max_vars][::-1]
    return feature_importance[sorted_idx], np.array(X_data.columns)[sorted_idx]
    
    
def calculate_permutation_feature_importances(
        X_data, y_data, model, max_vars: int|None = None, n_repeats: int = 10, random_state: int = 666):
    """ """
    feature_importance = permutation_importance(
        model, X_data, y_data, n_repeats = n_repeats, random_state = random_state)
    num_features = len(feature_importance.importances_mean)
    if not max_vars or max_vars >= num_features:
        max_vars = num_features
    sorted_idx = np.argsort(feature_importance.importances_mean)[::-1][:max_vars][::-1]
    return feature_importance.importances_mean[sorted_idx], np.array(X_data.columns)[sorted_idx]
    
    
def calculate_shap_feature_importances(X_data, model, max_vars: int|None = None):
    """ """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_data)
    shap_importance = shap_values.abs.mean(0).values
    num_features = len(shap_importance)
    if not max_vars or max_vars >= num_features:
        max_vars = num_features
    
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1][:max_vars][::-1]
    return shap_values, np.array(X_data.columns)[sorted_idx]
    
    
def model_feature_importances(model, X_data, y_data, feature_type: str, max_vars: int | None = None):
    """ """
    type_fun_map = {
        "gini" : partial(calculate_gini_feature_importances, X_data, max_vars = max_vars),
        "permutation": partial(calculate_permutation_feature_importances, X_data, y_data, max_vars = max_vars),
        "shap": partial(calculate_shap_feature_importances, X_data, max_vars = max_vars)
    }
    return type_fun_map[feature_type](model)


def plot_gini_feature_importances(
        model, X_data, max_vars: int|None = None, title: str|None = None, ax = None, tick_fontsize: int = 8,
        title_fontsize: int = 10, label_fontsize: int = 8) -> None:
    """Plots feature importances based on impurity loss, e.g. entropy or gini"""
    if ax is None:
        ax = plt.gca()
    if not title:
        title = "Feature Importances"
    feature_importance = model.feature_importances_
    num_features = len(feature_importance)
    if not max_vars or max_vars >= num_features:
        max_vars = num_features
    sorted_idx = np.argsort(feature_importance)[::-1][:max_vars][::-1]
    ax.barh(
        range(len(sorted_idx)), 
        feature_importance[sorted_idx], 
        align='center', color='turquoise', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(sorted_idx)), np.array(X_data.columns)[sorted_idx])
    ax.xaxis.set_tick_params(labelsize=tick_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize)


def plot_permutation_feature_importances(
        model, X_data, y_data, max_vars: int|None = None, title: str|None = None, 
        n_repeats: int = 10, random_state: int = 666, ax = None, tick_fontsize: int = 8,
        title_fontsize: int = 10, label_fontsize: int = 8) -> None:
    """Plots feature importances calculated as permutation importances"""
    if ax is None:
        ax = plt.gca()
    if not title:
        title = "Feature Importances"
    feature_importance = permutation_importance(
        model, X_data, y_data, n_repeats = n_repeats, random_state = random_state)
    num_features = len(feature_importance.importances_mean)
    if not max_vars or max_vars >= num_features:
        max_vars = num_features
    #sorted_idx = np.argsort(feature_importance)[:max_vars]
    sorted_idx = np.argsort(feature_importance.importances_mean)[::-1][:max_vars][::-1]
    #if max_vars:
    #    sorted_idx = sorted_idx[:max_vars]
    #fig = plt.figure(figsize=(12, 6))
    ax.barh(
        range(len(sorted_idx)), 
        feature_importance.importances_mean[sorted_idx], 
        align='center', color='turquoise', edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(sorted_idx)), np.array(X_data.columns)[sorted_idx])
    ax.xaxis.set_tick_params(labelsize=tick_fontsize)
    ax.yaxis.set_tick_params(labelsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize)


def plot_shap_feature_importances(
        model, X_data, y_data = None, max_vars: int|None = None, title: str|None = None,
        plot_type: Literal['shap', 'bar'] = 'bar', ax = None, tick_fontsize: int = 8,
        title_fontsize: int = 10, label_fontsize: int = 8, clustering: bool = False) -> None:
    """Plots feature importances as mean shap values"""
    if ax is None:
        ax = plt.gca()
    if not title:
        title = "Feature Importances"
    explainer = shap.Explainer(model)
    shap_values = explainer(X_data)
    shap_importance = shap_values.abs.mean(0).values
    num_features = len(shap_importance)
    if not max_vars or max_vars >= num_features:
        max_vars = num_features
    sorted_idx = shap_importance.argsort()[::-1][:max_vars][::-1]
    if plot_type == 'bar':
        ax.barh(
            range(len(sorted_idx)), 
            shap_importance[sorted_idx], 
            align='center', color='#80B1D3', edgecolor='black', linewidth=1, alpha = 0.55)
        ax.set_yticks(range(len(sorted_idx)), np.array(X_data.columns)[sorted_idx])
        ax.xaxis.set_tick_params(labelsize=tick_fontsize)
        ax.yaxis.set_tick_params(labelsize=tick_fontsize)
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel('mean(|SHAP Value|)', fontsize=label_fontsize)
        ax.axes.get_yaxis().set_visible(True)
        ax.axes.get_xaxis().set_visible(True)
        ax.spines["bottom"].set_color("black")      # x axis line
    if plot_type == "shap":
        if clustering and y_data is not None:
            shap_clusters = shap.utils.hclust(X_data, y_data) 
            shap.plots.bar(shap_values, max_display=max_vars, clustering=shap_clusters)
        else:
            shap.plots.bar(shap_values, max_display=max_vars)
        ax.set_title(title, fontsize=title_fontsize)
        #fig.suptitle(title)


def remove_outliers(X, y, num_neighbours: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Uses LocalOutlierFactor method to filter out outliers in the features dataset"""
    lof = LocalOutlierFactor(n_neighbors = num_neighbours)
    yhat = lof.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    if not all(mask):
        print(f'Removed {sum(mask==False)} reservoir(s): ')
        print(", ". join(list(y['Reservoir'][~mask]))) # ", ".join(*list(y[~mask]))
        X = X[mask]
        y = y[mask]
    return X, y
