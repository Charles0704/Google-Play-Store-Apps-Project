import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split, cross_val_score
from Metrics import cm, ROC
# Classification
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgbm
import catboost as cat
from catboost import CatBoostClassifier, Pool
from sklearn import manifold, datasets
import optuna
import warnings
warnings.filterwarnings("ignore")
import os

class Model:
    def __init__(self, model, n_trials):
        self.model = model
        self.n_trials = n_trials
        
    def preprocess_data(self):
        project_root = os.path.abspath(os.path.join(os.getcwd()))
        self.data = pd.read_csv(os.path.join(project_root, "data_db.csv"))
        features = self.data.drop(['Rating', 'Rating Interval'], axis=1)
        label = self.data['Rating Interval']
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(features, label, test_size=0.25, random_state=413)
        return features, label

    def train(self, objective,if_imp=False, show_progress_bar=True):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, self.train_X, self.train_Y), n_trials=self.n_trials, show_progress_bar=show_progress_bar)
        best_params = study.best_params
        self.clf = self.model(**best_params)
        self.clf = self.clf.fit(self.train_X, self.train_Y)
        y_pred = self.clf.predict(self.test_X)
    
        if if_imp:
            importances = self.clf.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            feat_labels = self.train_X.columns
            for f in range(self.train_X.shape[1]):
                print("%2d) %-*s %f" % (f + 1, 10,
                                        feat_labels[sorted_indices[f]],
                                        importances[sorted_indices[f]]))
        return best_params, y_pred

    def metrics(self):
        cm(self.test_X, self.test_Y, self.clf, str(self.model), ['0', '1'])

def xgb_objective(trial, features, label):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
    }
    model = XGBClassifier(**params)
    model.fit(features, label)
    k_folds = 3
    cv_scores = cross_val_score(model, features, label, cv=k_folds, scoring='accuracy')
    return np.mean(cv_scores)

def lgbm_objective(trial, features, label):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'rf', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    model = lgbm.LGBMClassifier(**params)
    model.fit(features, label)
    k_folds = 3
    cv_scores = cross_val_score(model, features, label, cv=k_folds, scoring='accuracy')
    return np.mean(cv_scores)

def cat_objective(trial, features, label):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 0.1),
        'task_type': trial.suggest_categorical('task_type', ['GPU']),
        'loss_function': trial.suggest_categorical('loss_function', ['MultiClass'])
    }
    model = cat.CatBoostClassifier(**params)
    model.fit(features, label)
    k_folds = 3
    cv_scores = cross_val_score(model, features, label, cv=k_folds, scoring='accuracy')
    return np.mean(cv_scores)

def main(args):
    if args.model == "XGB":
        tree_model = Model(XGBClassifier, args.n_trials)
        features, label = tree_model.preprocess_data()
        A, y_pred_XGB = tree_model.train(xgb_objective, args.if_imp, args.show_progress_bar)
        tree_model.metrics()
    elif args.model == "lgbm":
        tree_model = Model(lgbm.LGBMClassifier, args.n_trials)
        features, label = tree_model.preprocess_data()
        A, y_pred_lgbm = tree_model.train(lgbm_objective, args.if_imp, args.show_progress_bar)
        tree_model.metrics()
    else:
        tree_model = Model(cat.CatBoostClassifier, args.n_trials)
        features, label = tree_model.preprocess_data()
        A, y_pred_cat = tree_model.train(cat_objective, args.if_imp, args.show_progress_bar)
        tree_model.metrics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', type=str, default="cat")
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--if_imp', type=bool, default=False)
    parser.add_argument('--show_progress_bar', type=bool, default=True)
    opt = parser.parse_args()
    main(opt)