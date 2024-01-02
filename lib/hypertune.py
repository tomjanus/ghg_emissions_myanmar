"""Hyperparameter tuning procedure for tree-based regression models in 
https://www.kaggle.com/code/bigironsphere/parameter-tuning-in-one-function-with-hyperopt
converted to a class-based format

Implements hyperparameter tuning in hyperopt (alternative package for hyperparameter tuning is optuna)
"""
import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Literal, Optional, List, Any
from enum import Enum
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import gc
import numpy as np
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from .utils import load_model, save_model
import warnings
warnings.filterwarnings('ignore')

#OPTIONAL OUTPUT
BEST_SCORE = 0


@dataclass
class LightGBMRegressionHyperTune:
    """ """
    LGBM_MAX_LEAVES: ClassVar = 2**11  # maximum number of leaves per tree for LightGBM
    LGBM_MAX_DEPTH: ClassVar = 25  # maximum tree depth for LightGBM
    eval_metric: Literal['mae', 'rmse'] = "mae"
    num_evals: int = 100
    n_folds: int = 5

    def run(self, data, labels, diagnostic=False):
        """ """
        print('Running {} rounds of LightGBM parameter optimisation:'.format(self.num_evals))
        gc.collect()
        
        integer_params = [
            'max_depth', 'num_leaves', 'min_data_in_leaf'] #'max_bin' 'min_data_in_bin'
        
        def objective(space_params: Dict) -> Dict:
            # cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
            # extract nested conditional parameters
            if space_params['boosting']['boosting'] == 'goss':
                top_rate = space_params['boosting'].get('top_rate')
                other_rate = space_params['boosting'].get('other_rate')
                #0 <= top_rate + other_rate <= 1
                top_rate = max(top_rate, 0)
                top_rate = min(top_rate, 0.5)
                other_rate = max(other_rate, 0)
                other_rate = min(other_rate, 0.5)
                space_params['top_rate'] = top_rate
                space_params['other_rate'] = other_rate
            
            subsample = space_params['boosting'].get('subsample', 1.0)
            space_params['boosting'] = space_params['boosting']['boosting']
            space_params['subsample'] = subsample
            
            cv_results = lgb.cv(
                space_params, train, nfold = self.n_folds, stratified=False,
                metrics=self.eval_metric, seed=42)
            
            if self.eval_metric == "mae":
                best_loss = cv_results['valid l1-mean'][-1] 
            else:
                best_loss = cv_results['valid l2-mean'][-1] 
            return {'loss' : best_loss, 'status' : STATUS_OK }
        
        train = lgb.Dataset(data, labels)
        #integer and string parameters, used with hp.choice()
        boosting_list = [{'boosting': 'gbdt',
                          'subsample': hp.uniform('subsample', 0.5, 1)},
                         {'boosting': 'goss',
                          'subsample': 1.0,
                         'top_rate': hp.uniform('top_rate', 0, 0.5),
                         'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['MAE', 'RMSE'] 
        objective_list = ['huber', 'gamma', 'fair', 'tweedie']

        space ={'boosting' : hp.choice('boosting', boosting_list),
                'num_leaves' : hp.quniform('num_leaves', 2, self.LGBM_MAX_LEAVES, 1),
                #'max_bin': hp.quniform('max_bin', 32, 255, 1), # These commented out lines cause problems. Look into them when training models with categorical data
                'max_depth': hp.quniform('max_depth', 2, self.LGBM_MAX_DEPTH, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
                #'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
                'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),
                'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
                'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'metric' : hp.choice('metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
                'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01),
                'verbose': -1,
                'feature_pre_filter' : False
            }
        
        #optional: activate GPU for LightGBM
        #follow compilation steps here:
        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
        #then uncomment lines below:
        #space['device'] = 'gpu'
        #space['gpu_platform_id'] = 0,
        #space['gpu_device_id'] =  0

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.num_evals, 
                    trials=trials)
                
        #fmin() will return the index of values chosen from the lists/arrays in 'space'
        #to obtain actual values, index values are used to subset the original lists/arrays
        best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
        best['metric'] = metric_list[best['metric']]
        best['objective'] = objective_list[best['objective']]
                
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        if diagnostic:
            return best, trials
        else:
            return best


@dataclass
class XGBoostHyperTune:
    """ """
    XGB_MAX_LEAVES: ClassVar = 2**12  # maximum number of leaves when using histogram splitting
    XGB_MAX_DEPTH: ClassVar = 25  # maximum tree depth for XGBoost
    eval_metric: Literal['mae', 'rmse'] = "mae"
    num_evals: int = 100
    n_folds: int = 5
    with_gpu: bool = False

    def run(self, data, labels, diagnostic=False):
        """ """
        print('Running {} rounds of XGBoost parameter optimisation:'.format(self.num_evals))
        gc.collect()
        integer_params = ['max_depth']
        
        def objective(space_params: Dict) -> Dict:
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
            #extract multiple nested tree_method conditional parameters
            #libera te tutemet ex inferis
            if space_params['tree_method']['tree_method'] == 'hist':
                max_bin = space_params['tree_method'].get('max_bin')
                space_params['max_bin'] = int(max_bin)
                if space_params['tree_method']['grow_policy']['grow_policy']['grow_policy'] == 'depthwise':
                    grow_policy = space_params['tree_method'].get('grow_policy').get('grow_policy').get('grow_policy')
                    space_params['grow_policy'] = grow_policy
                    space_params['tree_method'] = 'hist'
                else:
                    max_leaves = space_params['tree_method']['grow_policy']['grow_policy'].get('max_leaves')
                    space_params['grow_policy'] = 'lossguide'
                    space_params['max_leaves'] = int(max_leaves)
                    space_params['tree_method'] = 'hist'
            else:
                space_params['tree_method'] = space_params['tree_method'].get('tree_method')
                
            cv_results = xgb.cv(
                space_params, train, nfold=self.n_folds, metrics=[self.eval_metric],
                early_stopping_rounds=100, stratified=False, seed=42)
            
            if self.eval_metric == "mae":
                best_loss = cv_results['test-mae-mean'].iloc[-1] 
            else:
                best_loss = cv_results['test-rmse-mean'].iloc[-1]
            return {'loss' : best_loss, 'status' : STATUS_OK }
        
        train = xgb.DMatrix(data, labels)
        
        #integer and string parameters, used with hp.choice()
        boosting_list = ['gbtree', 'gblinear'] #if including 'dart', make sure to set 'n_estimators'
        metric_list = ['mae', 'rmse', 'mape'] 
        
        if not self.with_gpu:
            tree_method = [
                {'tree_method' : 'exact'},
                {'tree_method' : 'approx'},
                {
                    'tree_method' : 'hist',
                    'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                    'grow_policy' : {
                        'grow_policy': {'grow_policy':'depthwise'},
                        'grow_policy' : {
                            'grow_policy':'lossguide',
                            'max_leaves': hp.quniform('max_leaves', 32, self.XGB_MAX_LEAVES, 1)
                        }
                    }
                }
            ]
        else:
            tree_method = [
                {'tree_method' : 'gpu_exact'},
                {'tree_method' : 'approx'},
                {
                    'tree_method' : 'gpu_hist',
                    'max_bin': hp.quniform('max_bin', 2**3, 2**7, 1),
                    'grow_policy' : {
                        'grow_policy': {'grow_policy':'depthwise'},
                        'grow_policy' : {
                            'grow_policy':'lossguide',
                            'max_leaves': hp.quniform('max_leaves', 32, self.XGB_MAX_LEAVES, 1)
                        }
                    }
                }
            ]
        
        objective_list = ['reg:linear', 'reg:gamma', 'reg:tweedie']
        
        space ={'boosting' : hp.choice('boosting', boosting_list),
                'tree_method' : hp.choice('tree_method', tree_method),
                'max_depth': hp.quniform('max_depth', 2, self.XGB_MAX_DEPTH, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
                'min_child_weight' : hp.uniform('min_child_weight', 0, 5),
                'gamma' : hp.uniform('gamma', 0, 5),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'eval_metric' : hp.choice('eval_metric', metric_list),
                'objective' : hp.choice('objective', objective_list),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'colsample_bynode' : hp.quniform('colsample_bynode', 0.1, 1, 0.01),
                'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                'nthread' : -1
            }
        
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.num_evals, 
                    trials=trials)
        
        best['tree_method'] = tree_method[best['tree_method']]['tree_method']
        best['boosting'] = boosting_list[best['boosting']]
        best['eval_metric'] = metric_list[best['eval_metric']]
        best['objective'] = objective_list[best['objective']]
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        if 'max_bin' in best:
            best['max_bin'] = int(best['max_bin'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return best, trials
        else:
            return best


@dataclass
class CatBoostHyperTune:
    """ """
    CB_MAX_DEPTH: ClassVar = 8  # maximum tree depth in CatBoost
    OBJECTIVE_CB_REG: ClassVar = 'MAE'  # CatBoost regression metric
    eval_metric: Literal['mae', 'rmse'] = "mae"
    cat_features: Optional[List[str]] = None
    num_evals: int = 100
    n_folds: int = 5
    with_gpu: bool = False
    best: Any | None = None

    def run(self, data, labels, diagnostic=False):
        """ """

        print('Running {} rounds of CatBoost parameter optimisation:'.format(self.num_evals))
        gc.collect()
            
        integer_params = ['depth',
                          'one_hot_max_size', #for categorical data
                          'min_data_in_leaf',
                          'max_bin']
        
        def objective(space_params):
            #cast integer params from float to int
            for param in integer_params:
                try:
                    space_params[param] = int(space_params[param])
                except KeyError:
                    pass
            #extract nested conditional parameters
            if space_params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
                bagging_temp = space_params['bootstrap_type'].get('bagging_temperature')
                space_params['bagging_temperature'] = bagging_temp
                
            if space_params['grow_policy']['grow_policy'] == 'LossGuide':
                max_leaves = space_params['grow_policy'].get('max_leaves')
                space_params['max_leaves'] = int(max_leaves)
                
            space_params['bootstrap_type'] = space_params['bootstrap_type']['bootstrap_type']
            space_params['grow_policy'] = space_params['grow_policy']['grow_policy']
                           
            #random_strength cannot be < 0
            space_params['random_strength'] = max(space_params['random_strength'], 0)
            #fold_len_multiplier cannot be < 1
            space_params['fold_len_multiplier'] = max(space_params['fold_len_multiplier'], 1)
                       
            #for classification set stratified=True
            cv_results = cb.cv(
                train, space_params, fold_count=self.n_folds, 
                early_stopping_rounds=25, stratified=False, partition_random_seed=42)
           
            if self.eval_metric == "mae":
                best_loss = cv_results['test-MAE-mean'].iloc[-1]
            else:
                best_loss = cv_results['test-RMSE-mean'].iloc[-1]
            return {'loss' : best_loss, 'status' : STATUS_OK}
        
        train = cb.Pool(data, labels.astype('float32'), cat_features=self.cat_features)
        
        #integer and string parameters, used with hp.choice()
        if self.with_gpu:
            bootstrap_type = [{'bootstrap_type':'Poisson'}]
        else:
            bootstrap_type = []
        
        bootstrap_type.extend(
            [{
                'bootstrap_type':'Bayesian',
                'bagging_temperature' : hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
            {'bootstrap_type':'Bernoulli'}
            ])

        if self.with_gpu:
            LEB = ['No', 'AnyImprovement', 'Armijo']
        else:
            LEB = ['No', 'AnyImprovement']
        if not self.with_gpu:
            score_function = ['Cosine', 'L2']
        else:
            score_function = ['Cosine', 'L2', 'NewtonCorrelation', 'NewtonL2']
        grow_policy = [{'grow_policy':'SymmetricTree'},
                       {'grow_policy':'Depthwise'},
                       {'grow_policy':'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]
        eval_metric_list = ['MAE', 'RMSE', 'Poisson']
                
        space ={'depth': hp.quniform('depth', 2, self.CB_MAX_DEPTH, 1), # Important, can be set to 2 - 10
                'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254
                'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0, 5), # Important 1-10?  
                'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
                'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
                'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
                'learning_rate' : hp.uniform('learning_rate', 0.01, 0.25), # Important 1
                'eval_metric' : hp.choice('eval_metric', eval_metric_list),
                'objective' : self.OBJECTIVE_CB_REG,
                'score_function' : hp.choice('score_function', score_function), #crashes kernel - reason unknown
                'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
                'grow_policy': hp.choice('grow_policy', grow_policy),
                'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
                'od_type' : 'Iter',
                'od_wait' : 25,
                'task_type' : 'GPU',
                'verbose' : 0,
                'logging_level' : None
            }
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            # 'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 8)
        
        if self.cat_features is not None:
            space['one_hot_max_size'] = hp.quniform('one_hot_max_size', 2, 16, 1)

        if not self.with_gpu:
            space['task_type'] = 'CPU' 
            space['max_bin'] = hp.uniform('max_bin', 253, 254)
            space['colsample_bylevel'] = hp.quniform('colsample_bylevel', 0.1, 1, 0.01) # CPU only

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.num_evals, 
                    trials=trials)

        #unpack nested dicts first
        best['bootstrap_type'] = bootstrap_type[best['bootstrap_type']]['bootstrap_type']
        best['grow_policy'] = grow_policy[best['grow_policy']]['grow_policy']
        best['eval_metric'] = eval_metric_list[best['eval_metric']]
        best['score_function'] = score_function[best['score_function']] 
        if not self.with_gpu:
            # TODO: doesn't seem to be found in the list of hyperparameters
            #best['leaf_estimation_method'] = LEB[best['leaf_estimation_method']] #CPU only
            pass
        best['leaf_estimation_backtracking'] = LEB[best['leaf_estimation_backtracking']]        
        
        #cast floats of integer params to int
        for param in integer_params:
            try:
                best[param] = int(best[param])
            except KeyError:
                pass
        if 'max_leaves' in best:
            best['max_leaves'] = int(best['max_leaves'])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')

        self.best = best
        
        if diagnostic:
            return best, trials
        else:
            return best
            
            
class HyperTuner(Enum):
    LIGHTGBM = LightGBMRegressionHyperTune
    XGBOOST = XGBoostHyperTune
    CATBOOST = CatBoostHyperTune
    
    
def hypertune_model(
        X_data, y_data, num_evals: int, hypertuner: HyperTuner, file: str | None = None,
        override: bool = False):
    """Finds optimum values of hyperparameters with hypertune for a number for a number of
    regression models given in `tuner_regressor_map`"""
    tuner_regressor_map = {
        HyperTuner.LIGHTGBM: LGBMRegressor,
        HyperTuner.XGBOOST: XGBRegressor,
        HyperTuner.CATBOOST: CatBoostRegressor
    }
    if os.path.exists(file) and not override:
        # Use the trained model
        print("Trained model already exists, reading model from file")
        model = load_model(file)
        return model
    tuner = hypertuner.value(num_evals = num_evals)
    best_params = tuner.run(X_data, y_data)
    model = tuner_regressor_map[hypertuner](**best_params)
    print("Finding optimal values of hyperparameters...")
    model.fit(X_data, y_data)
    if file:
        print(f"Saving model to {file}")
        save_model(model=model, file=file)
    return model


if __name__ == "__main__":
    ...
