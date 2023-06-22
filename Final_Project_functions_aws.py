# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from pandas.core.dtypes.common import classes

import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.utils import resample
from scipy import stats
import multiprocessing
from IPython.display import display

# from sklearn.preprocessing import FunctionTransformer # we dont need this
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import mannwhitneyu,chi2_contingency
from sklearn.decomposition import PCA


#optuna
#!pip install optuna
import optuna
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn import model_selection
#from scikit_posthocs import posthoc_nemenyi_friedman

# XGBoost
#!pip install xgboost
from xgboost import XGBClassifier

def train_test_pipeline_models_f1score(X, y, model, numeric_features, categorical_features, bootstrap_index_from_func, PCA=False, var_exp=None):
    ### Initialize an empty deque (they are faster for appends and pops) for logLoss. Probably not necessary, but it's fun.
    f1score = collections.deque(maxlen=len(bootstrap_index_from_func))

    ### for each set of inbag/outbag in the list of them do something:
    for train_index, test_index in bootstrap_index_from_func:
        ### define X_train and X_test as the rows that have indicies matching those from the bootstraping
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

        ### define y_train and y_test as the rows that have indicies matching those from the bootstraping
        y_train, y_test = y[train_index], y[test_index]
        ### run the pipeline and fit on X and y train datasets
        if PCA:
            X_train_processed, X_test_processed = pca_with_split(X_train, X_test, numeric_features, categorical_features, var_exp)

        else:
            X_train_processed, X_test_processed = preprocessor(X_train, X_test, y_train, numeric_features, categorical_features)

        # Train the model

        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)


        ### append the logLoss between the predicted and the true data to the logLoss deque
        f1score.append(f1_score(y_test, y_pred, average='weighted'))

    ### Return the list of f1_weighted
    return (f1score)

def preprocessor(X_data_train, X_data_test, y_data_train, numeric_features, cat_features):
    num_transformed_X_train, num_transformed_X_test, _, numeric_cols = numeric_transformer(X_data_train, X_data_test, y_data_train, numeric_features)
    cat_transformed_X_train, cat_transformed_X_test, _, categorical_cols = categorical_transformer(X_data_train, X_data_test, y_data_train, cat_features)
    X_train_processed = num_transformed_X_train.join(cat_transformed_X_train, how='inner')
    X_test_processed = num_transformed_X_test.join(cat_transformed_X_test, how='inner')
    return(X_train_processed, X_test_processed)



def numeric_transformer(X_data_train, X_data_test, y_data_train, numeric_cols):
    # SimpleImputer drops columns that are all nans.  Not sure if KNNImputer does, but best be on the safe side.
    # In order to preserve which column is dropped so that the code will work we need to check for columns that are
    # all nans and then remove them
    # from our list of features (numeric_cols)
    # check for columns that are all nan
    features_to_drop = []
    for fe in numeric_cols:
        if X_data_train[fe].isnull().all():
            features_to_drop.append(fe)
    numeric_cols = [val for val in numeric_cols if val not in features_to_drop]
    ###################################
    ### Impute Missing Values
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, weights='distance')
    imputer.fit(X_data_train[numeric_cols])
    X_imp_tr = pd.DataFrame(imputer.transform(X_data_train[numeric_cols]),columns=numeric_cols)
    X_imp_test = pd.DataFrame(imputer.transform(X_data_test[numeric_cols]),columns=numeric_cols)

    ##################################
    ### Feature selection
    # Remove all insignificant numeric features from consideration
    #testvar=len(numeric_cols)
    newNumFeatures = []
    for fe in numeric_cols:
        _, p = mannwhitneyu(X_imp_tr[fe].to_numpy().flatten(), y_data_train.flatten())
        if (p <= 0.05):
            newNumFeatures.append(fe)
    numeric_cols = newNumFeatures
    #print('Removed ', testvar-len(numeric_cols), ' numeric features')
    ##################################
    ### Scaling

    scaler = StandardScaler()
    scaler.fit(X_imp_tr[numeric_cols])
    X_imp_tr[numeric_cols] = scaler.transform(X_imp_tr[numeric_cols])
    X_imp_test[numeric_cols] = scaler.transform(X_imp_test[numeric_cols])
    return(X_imp_tr, X_imp_test, y_data_train, numeric_cols)

def categorical_transformer(X_data_train, X_data_test, y_data_train, categorical_cols):
    # SimpleImputer drops columns that are all nans.  In order to preserve which column is dropped
    # so that the code will work we need to check for columns that are all nans and then remove them
    # from our list of features (categorical_cols)
    # check for columns that are all nan
    features_to_drop = []
    for fe in categorical_cols:
        if X_data_train[fe].isnull().all():
            features_to_drop.append(fe)
    categorical_cols = [val for val in categorical_cols if val not in features_to_drop]
    #################################
    ### Impute Missing Values
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(X_data_train[categorical_cols])
    X_imp_tr = pd.DataFrame(imputer.transform(X_data_train[categorical_cols]), columns=categorical_cols)
    X_imp_test = pd.DataFrame(imputer.transform(X_data_test[categorical_cols]), columns=categorical_cols)

    ##################################
    ### Feature selection
    # Remove all insignificant categorical features from consideration
    #testvar=len(categorical_cols)
    newCatFeatures = []
    for fe in categorical_cols:
        table = pd.crosstab(X_imp_tr[fe].to_numpy().flatten(), y_data_train.flatten())
        _, p, _, _ = chi2_contingency(table)
        if (p <= 0.05):
            newCatFeatures.append(fe)
    categorical_cols=newCatFeatures
    #print('Removed ',testvar-len(categorical_cols),' categorical features')
    ##################################
    ### Encoding

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_imp_tr[categorical_cols])
    columns_to_add = encoder.get_feature_names(categorical_cols)
    X_imp_tr_enc = pd.DataFrame.sparse.from_spmatrix(encoder.transform(X_imp_tr[categorical_cols]), columns=columns_to_add)
    X_imp_test_enc = pd.DataFrame.sparse.from_spmatrix(encoder.transform(X_imp_test[categorical_cols]), columns=columns_to_add)
    return(X_imp_tr_enc, X_imp_test_enc, y_data_train, categorical_cols)



def objective(trial, model_name, X, y, numeric_features, categorical_features, bootstraps_param):


    # Pick a model type
    # model_name = trial.suggest_categorical("model_type", ["DecisionTree","KNearestNeighbors","SVC","RandomForest","XGB"])
    # model_name = trial.suggest_categorical("model_type", ["XGB"])
    # Get the hyperparameters based on the model type
    PCA = trial.suggest_int('PCA',0,1)
    if PCA:
        var_exp = trial.suggest_int('PCA_Var_explained',70,98)*.01
    else:
        var_exp = None
    # DecisionTree
    if model_name == "DecisionTree":
        # Get the hyperparameters from a given range
        criterion_split = trial.suggest_categorical("dt_criterion_dt", ["gini", "entropy"])
        splitter_suggest = trial.suggest_categorical("splitter_dt", ["best", "random"])
        md = trial.suggest_int('max_depth_dt', 2, 10, 1)
        mi = trial.suggest_int('min_inst_dt', 1, 32, 1)
        # Define the model with the chosen parameters
        model = DecisionTreeClassifier(criterion=criterion_split, max_depth=md, min_samples_leaf=mi,
                                       splitter=splitter_suggest, )

    # KNN
    if model_name == "KNearestNeighbors":
        # Get the hyperparameters from a given range
        n_neighbors_select = trial.suggest_int("n_neighbors_knn", 3, 30, 1)
        weights_select = trial.suggest_categorical("weights_knn", ["uniform", "distance"])
        # Define the model with the chosen parameters
        model = KNeighborsClassifier(n_neighbors=n_neighbors_select, weights=weights_select)

    # SVC
    if model_name == "SVC":
        # Get the hyperparameters from a given range
        #kern = trial.suggest_categorical("SVC_kernel", ["linear", "poly", "rbf", "sigmoid"])
        kern = trial.suggest_categorical("SVC_kernel", ["linear", "sigmoid", "rbf"])
        C = trial.suggest_categorical("SVC_C", np.logspace(0.01, 2, 10, endpoint=True))
        if (kern == "poly"):
            deg = trial.suggest_int("SVC_degree", 2, 10, log=False)
        else:
            deg = 3
        # Define the model with the chosen parameters
        model = svm.SVC(kernel=kern, C=C, degree=deg, probability=True)

    # RandomForest
    if model_name == "RandomForest":
        # Get the hyperparameters from a given range
        criterion_split = trial.suggest_categorical("dt_criterion_rf", ["gini", "entropy"])
        n_estimators_suggest = trial.suggest_int("n_estimators_rf", 100, 200, 1)
        md = trial.suggest_int('max_depth_rf', 2, 10, 1)
        mi = trial.suggest_int('min_inst_rf', 1, 32, 1)

        model = RandomForestClassifier(n_estimators=n_estimators_suggest, criterion=criterion_split, max_depth=md,
                                       min_samples_leaf=mi)

        # Define the model with the chosen parameters

    if model_name == "XGB":
        # Get the hyperparameters for a given range
        param = {
            #'tree_method': trial.suggest_categorical('tree_method_xg', ['auto', 'exact', 'approx', 'hist']),
            #'tree_method': trial.suggest_categorical('tree_method_xg', ['auto', 'hist'])
	        #'booster': trial.suggest_categorical('boosters_xg', ['gbtree', 'gblinear', 'dart']),
            # this parameter means using the GPU when training our model to speedup the training process
            'lambda': trial.suggest_float('lambda_xg', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha_xg', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree_xg', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample_xg', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate_xg',
                                                       [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators': trial.suggest_int('n_estimators_xg', 100, 200, 1),
            'max_depth': trial.suggest_categorical('max_depth_sg', [5, 7, 9, 11, 13, 15, 17]),
            'random_state': trial.suggest_categorical('random_state_xg', [2020]),
            'min_child_weight': trial.suggest_int('min_child_weight_xg', 1, 300),
        }
        model = XGBClassifier(**param)

        # Get the negative mean squared error score with cross validation
    bootstrap_index_from_func = bootstrap_index_generator(X, bootstraps=bootstraps_param, stratify_target=y)

    ### get the logLoss from the train_test_pipeline_models_logLoss function.  See it's definition further below
    f1scores = train_test_pipeline_models_f1score(X, y, model, numeric_features, categorical_features, bootstrap_index_from_func, PCA, var_exp)
    # scores = Multiple_classifier_pipeline(X, y, model, 10, stratify_target=y)
    score = np.mean(f1scores)
    # Return the score
    return score


def bootstrap_index_generator(Dataset, bootstraps=1, stratify_target=None):
    bootstrap_indicies = []
    if not isinstance(bootstraps, int):
        print("Must pass a number of bootstraps as an integer")
        return
    else:
        for i in range(0,bootstraps):
            Index=range(0,len(Dataset))
            X_index_inbag=resample(Index, replace=True, n_samples=len(Dataset), random_state=i, stratify=stratify_target)
            X_inbag=[Index[element] for element in X_index_inbag]
            X_outbag=[Index[element] for element in Index if element not in X_index_inbag]
            bootstrap_indicies.append([X_inbag, X_outbag])
        return bootstrap_indicies

def pca_with_split(X_data_train, X_data_test, numeric_cols, categorical_cols, var_exp):
    features_to_drop = []
    for fe in numeric_cols:
        if X_data_train[fe].isnull().all():
            features_to_drop.append(fe)
    numeric_cols = [val for val in numeric_cols if val not in features_to_drop]
    # Need to run the imputations previous to PCA
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, weights='distance')
    imputer.fit(X_data_train[numeric_cols])
    X_imp_num_tr = pd.DataFrame(imputer.transform(X_data_train[numeric_cols]), columns=numeric_cols)
    X_imp_num_test = pd.DataFrame(imputer.transform(X_data_test[numeric_cols]), columns=numeric_cols)

    features_to_drop = []
    for fe in categorical_cols:
        if X_data_train[fe].isnull().all():
            features_to_drop.append(fe)
    categorical_cols = [val for val in categorical_cols if val not in features_to_drop]
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(X_data_train[categorical_cols])
    X_imp_cat_tr = pd.DataFrame(imputer.transform(X_data_train[categorical_cols]), columns=categorical_cols)
    X_imp_cat_test = pd.DataFrame(imputer.transform(X_data_test[categorical_cols]), columns=categorical_cols)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_imp_cat_tr[categorical_cols])
    columns_to_add = encoder.get_feature_names(categorical_cols)
    X_imp_cat_tr_enc = pd.DataFrame.sparse.from_spmatrix(encoder.transform(X_imp_cat_tr[categorical_cols]),
                                                     columns=columns_to_add)
    X_imp_cat_test_enc = pd.DataFrame.sparse.from_spmatrix(encoder.transform(X_imp_cat_test[categorical_cols]),
                                                       columns=columns_to_add)

    # X_train = X_imp_num_tr.join(X_imp_cat_tr_enc, how='inner')
    X_train = X_imp_num_tr
    # X_test = X_imp_num_test.join(X_imp_cat_test_enc, how='inner')
    X_test = X_imp_num_test

    pca=PCA()
    pca.fit(X_train)
    explained_var = pca.explained_variance_ratio_.cumsum()
    n_over_var_exp = len(explained_var[explained_var>= var_exp])
    n_to_reach_var_exp = X_train.shape[1]-n_over_var_exp +1
    pca=PCA(n_to_reach_var_exp)
    pca.fit(X_train)
    X_train_var_exp = pd.DataFrame(pca.transform(X_train))
    X_train_var_exp = X_train_var_exp.join(X_imp_cat_tr_enc, how='inner')

    X_test_var_exp = pd.DataFrame(pca.transform(X_test))
    X_test_var_exp = pca.transform(X_test).join(X_imp_cat_test_enc, how='inner')

    return(X_train_var_exp, X_test_var_exp)


