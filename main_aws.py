# Import our defined functions
from Final_Project_functions import *


#################################################################################################################################
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
X, X_valid, y, y_valid = train_test_split(X_all, y_all, test_size=.3, stratify=y_all)
Xt = X.reset_index().drop('index', axis = 1)
X_valid = X_valid.reset_index().drop('index', axis = 1)
yt = y.reset_index().drop('index', axis = 1)
y_valid= y_valid.reset_index().drop('index', axis = 1)

### Define the numeric and categorical columns.  Based on the dataset description.
num_cols = X.columns[list([0] + list(range(33,38))+[82]+list(range(84,90)))]
cat_cols = [name for name in X.columns if not(name in num_cols)]


model_type = ["DecisionTree","KNearestNeighbors","RandomForest","XGB"]
#model_type = ["SVC"]
#constants = [Xt, yt, num_cols, cat_cols]
#tasks=[]
#for i in (range(len(model_type))):
#    temp = constants
#    temp.insert(0,model_type[i])
#    tasks.append(temp)

#study_results_dict = {}
#def optuna_parallel_task(model):
#    study = optuna.create_study(direction="maximize")
#    func = lambda trial: objective(trial,model,Xt,yt.target.ravel(), num_cols, cat_cols)
#    study.optimize(func, n_trials=10)
#    study_results_dict[(f'studies_param_{model}')] = [study.best_trial.params,study.best_trial.values[0]]

#with multiprocessing.Pool(processes=3) as pool:
#    result = pool.imap(optuna_parallel_task, model_type)


# result = map(optuna_parallel_task, model_type)


study_results_dict = {}

for model in model_type:
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(trial,model,Xt,yt.target.ravel(), num_cols, cat_cols, bootstraps_param = 15)
    study.optimize(func, n_trials=100)
    study_results_dict[(f'studies_param_{model}')] = [study.best_trial.params,study.best_trial.values[0]]

best_val = 0
for key in study_results_dict.keys():
    if study_results_dict[key][1] > best_val:
        best_val = study_results_dict[key][1]
params = None
for key in study_results_dict.keys():
    if study_results_dict[key][1] == best_val:
        params = study_results_dict[key][0]


#from multiprocessing import Pool


#if __name__ == '__main__':
#    with multiprocessing.Pool(processes=3) as pool:
#         result = pool.imap(optuna_parallel_task, model_type)
