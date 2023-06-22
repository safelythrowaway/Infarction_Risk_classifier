from sklearn.metrics import f1_score,precision_score, recall_score,precision_recall_curve,  average_precision_score,roc_curve, auc, roc_auc_score,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score,cross_val_predict

Xp,Xp_valid = preprocessor(Xt,X_valid,yt.target.ravel(),num_cols,cat_cols)
params = { 'eval_metric' : ['merror', 'mlogloss'],'lambda': 0.5220003411914401, 'alpha': 0.20105631575684482, 'colsample_bytree': 0.6, 'subsample': 1.0, 'learning_rate': 0.014, 'n_estimators': 101, 'max_depth': 15, 'random_state': 2020, 'min_child_weight': 6}
model = XGBClassifier(**params)
eval_set = [(Xp, yt.target.ravel()), (Xp_valid, y_valid.target.ravel())]
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

# for train_index, cv_index in k_fold.split(np.zeros(len(Xt)) ,yt.target.ravel()):
#     params = {'eval_metric': ['merror', 'mlogloss'], 'lambda': 0.5220003411914401, 'alpha': 0.20105631575684482,
#               'colsample_bytree': 0.6, 'subsample': 1.0, 'learning_rate': 0.014, 'n_estimators': 101, 'max_depth': 15,
#               'random_state': 2020, 'min_child_weight': 6}
#     model = XGBClassifier(**params)
#     X_train_fold, X_cv_fold = Xp.iloc[train_index,:], Xp.iloc[cv_index,:]
#     y_train_fold, y_cv_fold = yt.iloc[train_index], yt.iloc[cv_index]
#     model.fit(X_train_fold, y_train_fold)
#     print(y_train_fold)
#     print(model.predict(X_train_fold))
#     f1_scoreTraining = f1_score(y_train_fold, model.predict(X_train_fold),average = "weighted" )
#     trainingScores.append(f1_scoreTraining)
#     predictionsBasedOnKFolds.loc[X_cv_fold.index] = model.predict(X_cv_fold)
#     f1_scoreCV = f1_score(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index,1], average = "weighted")
#     cvScores.append(f1_scoreCV)
#     print('Training weighted F1_score: ', f1_scoreTraining)
#     print('CV Log Loss: ', f1_scoreCV)

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