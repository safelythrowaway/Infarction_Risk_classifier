#!pip install shap
from Final_Project_functions import *
import pandas as pd
import numpy as np
import shap
import sklearn
from xgboost import XGBClassifier

### Read in Data
url = 'https://raw.githubusercontent.com/acissej16j/6015_jj_jj_jr/main/MI.data.csv'

# Columns 113 - 1 Complications and outcomes of myocardial infarction
headerList = ['ID', 'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK','IBS_POST', 'IBS_NASL', 'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 'nr11', 'nr01', 'nr02', 'nr03','nr04','nr07', 'nr08', 'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10', 'endocr_01', 'endocr_02', 'endocr_03','zab_leg-01', 'zab_leg_02', 'zab_leg_03',
             'zab_leg_04','zab_leg_06', 'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P', 'ritm_ecg_p_01', 'ritm_ecg_p02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07',
             'ritm_ecg_p_08','n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09',
              'n_p_ecg_p_10', 'n_p_ecg_p_11','n_p_ecg_p_12', 'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08', 'GIP0_K','K_BLOOD', 'GIPER_Na', 'Na_BLOOD','ALT_BLOOD','AST_BLOOD','KFK_BLOOD','L_BLOOD','ROE','TIME_B_S', 'R_AB__1_n', 'R-AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 'NA_R_1_n','NA_R_2_n',
              'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n', 'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n', 'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS' ]
df = pd.read_csv(url, names=headerList)


#################################################################################################################################
### Format Data

# replace ? with NaNs
df = df.replace('?', np.nan)

# 1st target, lethal outcome

### Shuffle the data
df = df.sample(frac=1, random_state = 42)


df = df.drop(['ID'], axis = 1)
df = df.rename(columns = {"LET_IS":"target"})

y_all = df.target
X_all = df.drop(['target'], axis = 1)

### Hold out data for validation
X, X_valid, y, y_valid = train_test_split(X_all, y_all, test_size=.3)
Xt = X.reset_index().drop('index', axis = 1)
X_valid = X_valid.reset_index().drop('index', axis = 1)
yt = y.reset_index().drop('index', axis = 1)
y_valid= y_valid.reset_index().drop('index', axis = 1)

### Define the numeric and categorical columns.  Based on the dataset description.
num_cols = X.columns[list([0] + list(range(33,38))+[82]+list(range(84,90)))]
cat_cols = [name for name in X.columns if not(name in num_cols)]


def preprocessor(X_data_train, X_data_test, y_data_train, numeric_features, cat_features):
    num_transformed_X_train, num_transformed_X_test, _, numeric_cols = numeric_transformer(X_data_train, X_data_test, y_data_train, numeric_features)
    cat_transformed_X_train, cat_transformed_X_test, _, categorical_cols = categorical_transformer(X_data_train, X_data_test, y_data_train, cat_features)
    X_train_processed = num_transformed_X_train.join(cat_transformed_X_train, how='inner')
    X_test_processed = num_transformed_X_test.join(cat_transformed_X_test, how='inner')
    return(X_train_processed, X_test_processed)

params = {'lambda_xg': 0.5220003411914401, 'alpha_xg': 0.20105631575684482, 'colsample_bytree_xg': 0.6, 'subsample_xg': 1.0, 'learning_rate_xg': 0.014, 'n_estimators_xg': 101, 'max_depth_sg': 15, 'random_state_xg': 2020, 'min_child_weight_xg': 6}
model = XGBClassifier(**params)
XP, Xp_valid = preprocessor(Xt, X_valid, yt.target.ravel(), num_cols, cat_cols)
model.fit(XP, yt.target.ravel())
yt_pred = model.predict(Xp_valid)


#X_adult,y_adult = shap.datasets.adult()
background = shap.maskers.Independent(XP, max_samples=100)

# compute SHAP values

shap_values = shap.TreeExplainer(model).shap_values(XP)
shap.summary_plot(shap_values, XP)
explainer = shap.TreeExplainer(model)
# set a display version of the data to use for plotting (has string values)
#shap_values.display_data = shap.datasets.adult(display=True)[0].values
shap.summary_plot(shap_values, XP)
shap.plots.bar(shap_values)


from sklearn.metrics import f1_score,precision_score, recall_score,precision_recall_curve,  average_precision_score,roc_curve, auc, roc_auc_score,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score,cross_val_predict

Xp,Xp_valid = preprocessor(Xt,X_valid,yt.target.ravel(),num_cols,cat_cols)
params = { 'eval_metric' : ['merror', 'mlogloss'],'lambda': 0.5220003411914401, 'alpha': 0.20105631575684482, 'colsample_bytree': 0.6, 'subsample': 1.0, 'learning_rate': 0.014, 'n_estimators': 101, 'max_depth': 15, 'random_state': 2020, 'min_child_weight': 6}
model = XGBClassifier(**params)
eval_set = [(Xp, yt.target.ravel()), (Xp_valid, y_valid)]
model.fit(Xp, yt.target.ravel(), eval_set=eval_set, verbose=False)
model.fit(Xp,yt.target.ravel())
y_pred = model.predict(Xp_valid)
precision_score(y_pred,y_valid, average='weighted')


#{'studies_param_DecisionTree': [{'PCA': 0, 'dt_criterion_dt': 'gini', 'splitter_dt': 'random', 'max_depth_dt': 8, 'min_inst_dt': 28}, 0.8573680618890563], 'studies_param_KNearestNeighbors': [{'PCA': 0, 'n_neighbors_knn': 12, 'weights_knn': 'distance'}, 0.8195643383408402], 'studies_param_SVC': [{'PCA': 0, 'SVC_kernel': 'rbf', 'SVC_C': 2.8328411489680256}, 0.8591239128150615], 'studies_param_RandomForest': [{'PCA': 0, 'dt_criterion_rf': 'gini', 'n_estimators_rf': 200, 'max_depth_rf': 10, 'min_inst_rf': 2}, 0.8594386952049512], 'studies_param_XGB': [{'PCA': 0, 'lambda_xg': 0.714640600447979, 'alpha_xg': 0.5182096798473296, 'colsample_bytree_xg': 0.7, 'subsample_xg': 1.0, 'learning_rate_xg': 0.016, 'n_estimators_xg': 200, 'max_depth_sg': 5, 'random_state_xg': 2020, 'min_child_weight_xg': 9}, 0.8629263574969209]}



trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],index=yt.index,columns=[0,1])

k_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
results = cross_val_score(model, XP, yt, cv=k_fold)
print('Accuracy: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))

for train_index, cv_index in k_fold.split(np.zeros(len(Xt)) ,yt.target.ravel()):
    X_train_fold, X_cv_fold = Xp.iloc[train_index,:], Xp.iloc[cv_index,:]
    y_train_fold, y_cv_fold = yt.iloc[train_index], yt.iloc[cv_index]
    model.fit(Xp, yt.target.ravel())
    #print(y_train_fold)
    #print(model.predict_proba(X_train_fold)[:,1])
    f1_scoreTraining = f1_score(y_train_fold, model.predict_proba(X_train_fold)[:,1])
    trainingScores.append(f1_scoreTraining)
    predictionsBasedOnKFolds.loc[X_cv_fold.index,:] = model.predict_proba(X_cv_fold)
    f1_scoreCV = f1_score(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index,1])
    cvScores.append(f1_scoreCV)
    print('Training F1_score: ', f1_scoreTraining)
    print('CV Log Loss: ', f1_scoreCV)

results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = pyplot.subplots(figsize=(12, 12))
ax.plot(x_axis, results['validation_0']['mlogloss'], label ='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
pyplot.ylabel('LogLoss')
pyplot.title('XGBoostLogLoss')
pyplot.show()

### PR-curve
preds = pd.concat([yt,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],preds['prediction'])
average_precision = average_precision_score(preds['trueLabel'],preds['prediction'])
plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))


from matplotlib import pyplot