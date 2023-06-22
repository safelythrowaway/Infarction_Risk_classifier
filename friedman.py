from Final_Project_functions import *
from scikit_posthocs import posthoc_nemenyi_friedman
params = {'studies_param_DecisionTree': [{'PCA': 0, 'criterion': 'gini', 'splitter': 'random', 'max_depth': 8, 'min_samples_leaf': 28}, 0.8573680618890563], 'studies_param_KNearestNeighbors': [{'PCA': 0, 'n_neighbors': 12, 'weights': 'distance'}, 0.8195643383408402], 'studies_param_SVC': [{'PCA': 0, 'kernel': 'rbf', 'C': 2.8328411489680256}, 0.8591239128150615], 'studies_param_RandomForest': [{'PCA': 0, 'criterion': 'gini', 'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 2}, 0.8594386952049512], 'studies_param_XGB': [{'PCA': 0, 'lambda': 0.714640600447979, 'alpha': 0.5182096798473296, 'colsample_bytree': 0.7, 'subsample': 1.0, 'learning_rate': 0.016, 'n_estimators': 200, 'max_depth': 5, 'random_state': 2020, 'min_child_weight': 9}, 0.8629263574969209]}

model_type = ["DecisionTree","KNearestNeighbors","SVC","RandomForest","XGB"]

def Freid_eval(X, y, model_type, params, numeric_features, categorical_features):
    scores_dict = {}
    for model in model_type:
        model_params = params[f'studies_param_{model}'][0]
        del model_params['PCA']
        if model == "DecisionTree":
            model_running = DecisionTreeClassifier(**model_params)
        if model == "KNearestNeighbors":
            model_running = KNeighborsClassifier(**model_params)
        if model == "SVC":
            model_running = SVC(**model_params)
        if model == "RandomForest":
            model_running = RandomForestClassifier(**model_params)
        if model == "XGB":
            model_running = XGBClassifier(**model_params)

        bootstrap_index_from_func = bootstrap_index_generator(X, bootstraps=50, stratify_target=y)
        f1scores = train_test_pipeline_models_f1score(X, y, model_running, numeric_features, categorical_features,bootstrap_index_from_func)
        scores_dict[model] = f1scores
    return(scores_dict)

results = Freid_eval(Xt, yt.target.ravel(), model_type, params, num_cols, cat_cols)

test_significance=[]
for nameClassifier in results.keys():
    test_significance.append(results[nameClassifier])

#print(*test_significance)
#Check if Friedman test is signifiant
chi_square,p_value_mean=stats.friedmanchisquare(*test_significance)
print(f'Friedman Chi Square test p value mean: {p_value_mean}')

test_significance=[]
for nameClassifier in results.keys():
    test_significance.append(results[nameClassifier])

# If a significant difference exists, we can check for pairwise significant differences
chi_square,p_value_mean=stats.friedmanchisquare(*test_significance)
print(p_value_mean)

trans_groups=np.array(test_significance).T
print(trans_groups)
p=posthoc_nemenyi_friedman(trans_groups)
print(p)