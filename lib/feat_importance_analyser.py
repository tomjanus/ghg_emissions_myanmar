"""A small utility for analysing feature importance on tabular data

The data is first approximated with a random forest regressor. Then, feature
importance is calculated using various methods by querying the fitted random
forest model

The random forest can be tuned with fixed set of (hyper)parameters or following
a hyperparameter tuning procedure.

Introduces feature importance estimation with the following methods:
- built-in method in the RandomForest algorithm in sklearn
- permutation method
- using SHAP values

Built-in feature importances:
1. Gini importance (or mean decrease impurity) / variance reduction, which is 
    computed from the Random Forest structure. The drawbacks of the method is the
    tendency to prefer (select as important) numerical features and categorical 
    features with high cardinality. What is more, in the case of correlated 
    features it can select one of the feature and neglect the importance of the 
    second one (which can lead to wrong conclusions).
2. Permutation based importance
3. SHAP values (SHapley Addittive exPlanations)
4. Treeinterpreter

T.Janus
tomasz.k.janus@gmail.com

TODO: Extend the tool to use other ensemble regression methods such as Xgboost,
    CatBoost, LightGBM, etc.
"""
from typing import List, Tuple, Dict, Any, Literal, Sequence
from abc import ABC, abstractmethod
import pathlib
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import shap
from treeinterpreter import treeinterpreter as ti
from matplotlib import pyplot as plt


@dataclass
class XYData:
    """ """
    x: pd.DataFrame
    y: pd.Series
    output_name: str | None = None

    @property
    def feature_names(self) -> Sequence:
        return self.x.columns


@dataclass
class DataSplitter:
    """ """
    data: XYData
    test_size: int = 0.25
    random_state: int = 12

    def split(self) -> Tuple[XYData, XYData]:
        """Split data into training set and testing set"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.x, self.data.y, test_size=self.test_size, 
            random_state=self.random_state)
        output_name = self.data.output_name
        return XYData(X_train, y_train, output_name), \
            XYData(X_test, y_test, output_name)


@dataclass
class RegressionTuner(ABC):
    """ """
    train_data: XYData
    _model: RandomForestRegressor = field(init=False)

    @abstractmethod
    def fit(self) -> None:
        ...

    @property
    def model(self) -> RandomForestRegressor | None:
        try:
            return self._model
        except AttributeError:
            logging.warning("Model not present. Run fit() and try again.")
        return None

    def predict(self, test_data: XYData) -> np.ndarray:
        """Prediction with the (fitted) regression model"""
        return self.model.predict(test_data.x)

    def mape(self, test_data: XYData) -> float:
        """Find mean absolute percentage error between actual and predicted values"""
        mape = mean_absolute_percentage_error(self.predict(test_data), test_data.y)
        logging.debug(f"MAPE: {mape:.4f}")


@dataclass
class RandomForestRegressionTuner(RegressionTuner):
    """Fits random forests to train data and tests the fit on test_data"""
    # Random forest regressor model parameters
    n_proc: int = -1
    max_leaf_nodes: int | None = None
    max_depth: int | None = None
    max_features: float | int | Literal['sqrt', 'log2'] = 1
    n_estimators: int = 100
    random_state: int | None = None

    def fit(self) -> None:
        """ """
        model = RandomForestRegressor( 
            n_jobs=self.n_proc,
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes, 
            max_features=self.max_features, 
            random_state=self.random_state,
        )
        logging.info("fitting random forest ...")
        model.fit(self.train_data.x, self.train_data.y)
        self._model = model


@dataclass
class RandomForestRegressionHyperTuner(RegressionTuner):
    """ """
    hypertune_strategy: GridSearchCV | RandomizedSearchCV
    param_grid: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.param_grid:
            self.param_grid = {}

    def fit(self) -> None:
        """ """
        grid_search = self.hypertune_strategy(
            RandomForestRegressor(), self.param_grid)
        logging.info("fitting random forest with hyperparameter tuning ...")
        grid_search.fit(self.train_data.x, self.train_data.y)
        self._model = grid_search.best_estimator_


@dataclass
class FeatureImportanceCalculator(ABC):
    """ """
    tuner: RegressionTuner
    feature_indices: np.array = field(init = False)
    feature_names: Sequence = field(init=False)
    feature_importances: Sequence = field(init = False)

    @staticmethod
    def _sort(feature_importances: Sequence[float], num_features: int) -> np.ndarray:
        """Find indices of sorted and trimmed featurs"""
        sorted_idx = np.argsort(feature_importances)[-(num_features-1):]
        return sorted_idx

    @abstractmethod
    def calculate(self) -> None:
        ...

    @abstractmethod
    def plot(self) -> None:
        ...


@dataclass
class BuiltInFeatureImportanceCalculator(FeatureImportanceCalculator):
    """Finds feature importance using RandomForest regression implemented in
    ScikitLearn (sklearn)"""
    sort: bool = True
    num_features: int = 10

    def calculate(self) -> None:
        """ """
        feature_importances = self.tuner.model.feature_importances_

        if self.sort:
            sorted_idx = self._sort(
                feature_importances=feature_importances, 
                num_features=self.num_features)
            feature_importances = feature_importances[sorted_idx]
            self.feature_indices = sorted_idx
            self.feature_names = self.tuner.train_data.feature_names[sorted_idx]
        else:
            unsorted_idx = range(0,len(feature_importances))
            self.feature_indices = unsorted_idx
            self.feature_names = self.tuner.train_data.feature_names
            
        self.feature_importances = feature_importances

 
    def plot(self) -> None:
        """Display the calculated feature importances"""
        try:
            plt.barh(
                self.feature_names,
                self.feature_importances, color='b', align='center')
        except AttributeError:
            logging.warning("Feature importances not available. Run calculate() and try again.")
            return
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        plt.show()
            
    
@dataclass
class PermutationFeatureImportanceCalculator(FeatureImportanceCalculator):
    """Finds feature importance using permutation method in ScikitLearn (sklearn)"""
    data: XYData # Data against which the permutation feature importance is calculated against
                 # Normally test data
    sort: bool = True
    num_features: int = 10
    
    def calculate(self) -> None:
        """ """
        feature_importances_all = permutation_importance(
            self.tuner.model, self.test_data.x, self.test_data.y)
        feature_importances = feature_importances_all.importances_mean

        if self.sort:
            sorted_idx = self._sort(
                feature_importances=feature_importances, 
                num_features=self.num_features)
            feature_importances = feature_importances[sorted_idx]
            self.feature_indices = sorted_idx
            self.feature_names = self.tuner.train_data.feature_names[sorted_idx]
        else:
            unsorted_idx = range(0,len(feature_importances))
            self.feature_indices = unsorted_idx
            self.feature_names = self.tuner.train_data.feature_names
            
        self.feature_importances = feature_importances
        
    def plot(self) -> None:
        """Display the calculated feature importances"""
        try:
            plt.barh(
                self.feature_names,
                self.feature_importances, color='b', align='center')
        except AttributeError:
            logging.warning("Feature importances not available. Run calculate() and try again.")
            return
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        plt.show()
        

@dataclass
class SHAPFeatureImportanceCalculator(FeatureImportanceCalculator):
    """ """
    data: XYData # Data against which the shap values are calculated. Normally test data
    num_features: int = 10
    plot_type: Literal['bar', 'beeswarm', 'violin'] = 'bar'

    def calculate(self) -> None:
        explainer = shap.TreeExplainer(self.tuner.model)
        shap_values = explainer.shap_values(self.data.x)
        
        # Alternative method for calculating SHAP values
        # TODO: Investigate the differences between the current and this method
        # explainer = shap.Explainer(self.tuner.model.predict, self.data.x)
        # shap_values = explainer(self.data.x)
        # self.shap_values = shap_values
        
        self.shap_values = shap_values
        self.feature_importances = np.abs(shap_values).mean(axis=0)
        self.feature_indices = range(0, shap_values.shape[1])
        self.feature_names = self.tuner.train_data.feature_names
    
    def plot(self) -> None:
        """ Display shap values using various graphical representations """
        if self.plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                self.data.x,
                plot_type = "bar",
                max_display=self.num_features)
        elif self.plot_type == 'violin':
            shap.plots.violin(
                self.shap_values, 
                self.data.x,
                max_display=self.num_features)
        elif self.plot_type == 'beeswarm':
            shap.summary_plot(
                self.shap_values,
                self.data.x,
                plot_type = "violin",
                max_display=self.num_features)


@dataclass
class TreeIntFeatureImportanceCalculator(FeatureImportanceCalculator):
    """ """
    num_features: int = 10

    def calculate(self) -> None:
        """ """
        x_train = self.tuner.train_data.x
        prediction, bias, contributions = ti.predict(self.tuner.model, x_train)
        contributions = pd.Series(
            np.mean(contributions, axis=0), 
            index=x_train.columns)

        ix = self._sort(contributions.abs(), self.num_features+1)
        contributions_filtered = contributions[ix]
        self.contributions_filtered = contributions_filtered

        self.feature_importance = contributions_filtered
        self.feature_indices = np.array(None)
        self.feature_names = contributions_filtered.index
        
    def plot(self) -> None:
        """ """
        plt.barh(
            self.feature_names,
            self.feature_importance, color='g')
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        plt.show()


if __name__ == "__main__":
    fit_type: Literal["single", "hyper"] = "single"
    ft_importance_methods = ("treeinterpreter")

    # Load tabular data
    input_data = pd.read_csv(pathlib.Path("data/portfolio_options.csv"))
    output_name: str = 'Caister Terrestrial desalination'

    # Process categorical variables - important e.g. for visualising force plots
    # for data with cateforical (discrete) data present
    input_data = pd.get_dummies(input_data).fillna(input_data.mean())

    x = input_data.drop(output_name, axis=1)
    y = input_data[output_name]
         
    # Split data
    splitter = DataSplitter(XYData(x,y, output_name))
    train_data, test_data = splitter.split()

    if fit_type == "single":
        # Fit random forest
        random_forest_tuner = RandomForestRegressionTuner(
            train_data, max_depth=10, random_state=1, max_features=None)
        random_forest_tuner.fit()
        random_forest_tuner.predict(test_data)

        if "builtin" in ft_importance_methods:
            # Built in feature importance - OK
            ft_calculator_builtin = BuiltInFeatureImportanceCalculator(
                tuner=random_forest_tuner)
            ft_calculator_builtin.calculate()
            ft_calculator_builtin.plot()
        if "permutation" in ft_importance_methods:
            # Permutation-based feature importance
            ft_calculator_perm = PermutationFeatureImportanceCalculator(
                tuner=random_forest_tuner, data=test_data)
            ft_calculator_perm.calculate()
            ft_calculator_perm.plot()
        if "shap" in ft_importance_methods:
            ft_calculator_shap = SHAPFeatureImportanceCalculator(
                tuner=random_forest_tuner, data=test_data)
            ft_calculator_shap.calculate()
            ft_calculator_shap.plot() 
        if "treeinterpreter" in ft_importance_methods:
            ft_calculator_ti = TreeIntFeatureImportanceCalculator(
                tuner=random_forest_tuner)
            ft_calculator_ti.calculate()
            ft_calculator_ti.plot()

    # This part of the code tests fitting random forest regression with
    # different combinations of hyperparameters
    if fit_type == "hyper":
        # Set a dictionary of hyperparameters for parameter tuning
        param_grid: Dict[str, List[Any]] = {
            'n_jobs': [-1],
            'n_estimators': [25, 50, 100, 150],
            #'random_state': [1],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [6, 10, 14],
            'max_leaf_nodes': [3, 6],
        }
        random_forest_tuner_iter = RandomForestRegressionHyperTuner(
            train_data, hypertune_strategy=GridSearchCV, 
            param_grid = param_grid)
        random_forest_tuner_iter.fit()
        ft_calculator = BuiltInFeatureImportanceCalculator(
            tuner=random_forest_tuner_iter)
        ft_calculator.calculate()
        ft_calculator.plot()
    
