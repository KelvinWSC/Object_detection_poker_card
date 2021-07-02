#%% - Libraries import
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, recall_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

#%% - File reading
df = pd.read_csv('fordTrain.csv')
df.info()
df.describe()

#%% ================== Data cleansing section =======================
# corr = df.corr()
# index=[]
# value=[]
# for c in range(len(corr.columns)):
#     for r in range(len(corr.index)):
#         index.append(corr.columns[c]+'/'+corr.index[r])
#         value.append(corr.iloc[c,r])
# corr_rank=pd.DataFrame({'index':index,'value':value}).set_index('index')
# abs(corr_rank[corr_rank.value!=1].value).drop_duplicates().sort_values(ascending=False).head(10)

df = df.drop(['TrialID','ObsNum'], axis=1)
df = df.drop(['P8','V5','V7','V9'], axis=1) # drop useless columns with same values or skewed values
# df = df.drop(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P8', 'V7', 'V9', 'V6'], axis=1) # all p variables have low corr with target
df = df.drop(columns=['P3','V6','V10'])


#%% ========================== Train/Test split ==============================

x_train, x_test, y_train, y_test = train_test_split(df.drop('IsAlert', axis=1), df['IsAlert'], test_size=0.2, random_state=16)

#%%
# ============================== Data scaling ==============================

# Scaling the data for PCA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#%%
# ====================== logistic regression section ========================
logreg_pure = LogisticRegression()
logreg_pure.fit(x_train_scaled, y_train)
y_pred_log = logreg_pure.predict(x_test_scaled)

#%% - GridSearch for logistic regression
# penalty = ['l1', 'l2'] 
# C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# param_grid = dict(penalty=penalty, C=C)
# score = make_scorer(recall_score, pos_label=0,average='binary')
# grid = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring=score, verbose=1, n_jobs=-1) 
# grid_result = grid.fit(x_train_scaled, y_train)
# print('Best Score: ', grid_result.best_score_)
# print('Best Params: ', grid_result.best_params_)

#%% - logistic regression based on GridSearch result
logreg2 = LogisticRegression()
logreg2.fit(x_train_scaled, y_train)

y2_pred = logreg2.predict(x_test_scaled)
print(classification_report(y_test, y2_pred))


#%% - PCA
# ======================= PCA + logistic regression section =========================
pca = PCA(n_components=22)
x_train_pcalog = pca.fit_transform(x_train_scaled)
x_test_pcalog = pca.transform(x_test_scaled)

#%% - Logistic regression
logreg = LogisticRegression()
logreg.fit(x_train_pcalog, y_train)
y_train_pcalog_pred = logreg.predict(x_train_pcalog)
y_test_pcalog_pred = logreg.predict(x_test_pcalog)

print(classification_report(y_test, y_test_pcalog_pred))
# print(classification_report(y_train, y_train_pred))

# %% - GridSearch for logistic regression, for parameter C
score = make_scorer(recall_score, pos_label=0, average='binary')
grid = GridSearchCV(logreg, param_grid={'C':np.linspace(0.1,2,20)}, scoring=score, n_jobs=-1)
grid.fit(x_train_pcalog, y_train)
print(grid.best_score_)
print(grid.best_params_)

# %% - PCA + logistic regression based on GridSearch result
pca = PCA(n_components=22)
x_train_pcalog = pca.fit_transform(x_train_scaled)
x_test_pcalog = pca.transform(x_test_scaled)

logreg = LogisticRegression(C=0.1)
logreg.fit(x_train_pcalog, y_train)
y_train_pcalog_pred = logreg.predict(x_train_pcalog)
y_test_pcalog_pred = logreg.predict(x_test_pcalog)

print(classification_report(y_test, y_test_pcalog_pred))

# %% ====================== Random Forest section ========================
rnd_clf = RandomForestClassifier(n_estimators=1000, max_depth=60, min_samples_split=10, min_samples_leaf=1,n_jobs=-1, random_state=42)
rnd_clf.fit(x_train, y_train)

y_pred_rf = rnd_clf.predict(x_test)

print(classification_report(y_test,y_pred_rf))

# %% - check overfitting/underfitting of Random Forest

# Method 1 - Validation Curve

# param_range = range(10,100,10)
# train_scores, test_scores = validation_curve(RandomForestClassifier(), x_train, y_train, param_name="max_depth", param_range=param_range, scoring=score, n_jobs=-1, cv=3)
# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# Method 2 - Learning Curve

# train_sizes, train_scores_learning, validation_scores_learning = learning_curve(RandomForestClassifier(max_depth=20), X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3)
# train_scores_learning_mean = np.mean(train_scores_learning, axis=1)
# test_scores_learning_mean = np.mean(validation_scores_learning, axis=1)

# Visualization
# fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# axes[0].set_title("Validation Curve with Random Forest",fontsize=14)
# axes[0].set_xlabel('Max_depth',fontsize=10)
# axes[0].set_ylabel("Recall Score of 0",fontsize=10)
# axes[0].set_ylim(0.8, 1.05)
# axes[0].set_xlim(10,95)
# axes[0].plot(param_range, train_scores_mean, label="Training score",color="red",marker='o')
# axes[0].plot(param_range, test_scores_mean, label="Cross-validation score",color="navy",marker='x')
# axes[0].legend()

# axes[1].set_title("Learning Curve with Random Forest",fontsize=14)
# axes[1].set_xlabel('Training size',fontsize=10)
# axes[1].set_ylabel("Recall Score of 0",fontsize=10)
# axes[1].set_ylim(0.93, 1.0)
# axes[1].plot(train_sizes, train_scores_learning_mean, label="Training score",color="red", marker='o')
# axes[1].plot(train_sizes, test_scores_learning_mean, label="Cross-validation score",color="navy", marker='x')
# axes[1].legend(loc="best")
# plt.show()

# %% ========================== KNN section ==============================
k_range = range(1,5)
scores_knn = {}
scores_list = []

for k in k_range:
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train,y_train)
        y_pred_knn = knn.predict(x_test)
        scores_knn[k] = metrics.accuracy_score(y_test,y_pred_knn)
        scores_list.append(metrics.accuracy_score(y_test,y_pred_knn))

plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

# %% ========================== XGBoost section ==============================
xgmodel = xgboost.XGBClassifier()
xgmodel.fit(x_train, y_train)
y_pred_XG = xgmodel.predict(x_test)
y_pred_XG_train = xgmodel.predict(x_train)

print('Test result\n')
print(classification_report(y_test,y_pred_XG))
print('\nTrain result\n')
print(classification_report(y_train,y_pred_XG_train))

# %% - GridSearch for XGBoost, for parameter n_estimators
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 3)]
param_grid_a = {'n_estimators':n_estimators}

grid_search_a = GridSearchCV(xgmodel, param_grid = para_grid_a, n_jobs=-1, cv=3)
grid_search_a.fit(x_train,y_train)
grid_search_a.best_params_

# %% - check overfitting/underfitting of XGBoost

# param_range = range(10,100,10)
# train_scores_xgb, test_scores_xgb = validation_curve(RandomForestClassifier(), x_train, y_train, param_name="max_depth", param_range=param_range, scoring=score, n_jobs=-1,cv=3)
# train_scores_mean_xgb = np.mean(train_scores_xgb, axis=1)
# test_scores_mean_xgb = np.mean(test_scores_xgb, axis=1)

# train_sizes_xgb, train_scores_learning_xgb, validation_scores_learning_xgb = learning_curve(RandomForestClassifier(max_depth=20), x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3)
# train_scores_learning_mean_xgb = np.mean(train_scores_learning_xgb, axis=1)
# test_scores_learning_mean_xgb = np.mean(validation_scores_learning_xgb, axis=1)

# fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# axes[0].set_title("Validation Curve with XGBoost",fontsize=14)
# axes[0].set_xlabel('Max_depth',fontsize=10)
# axes[0].set_ylabel("Recall Score of 0",fontsize=10)
# axes[0].set_ylim(0.8, 1.05)
# axes[0].set_xlim(10,95)
# axes[0].plot(param_range, train_scores_mean_xgb, label="Training score",color="red",marker='o')
# axes[0].plot(param_range, test_scores_mean_xgb, label="Cross-validation score",color="navy",marker='x')
# axes[0].legend()

# axes[1].set_title("Learning Curve with XGBoost",fontsize=14)
# axes[1].set_xlabel('Training size',fontsize=10)
# axes[1].set_ylabel("Recall Score of 0",fontsize=10)
# axes[1].set_ylim(0.93, 1.0)
# axes[1].plot(train_sizes, train_scores_learning_mean_xgb, label="Training score",color="red", marker='o')
# axes[1].plot(train_sizes, test_scores_learning_mean, label="Cross-validation score",color="navy", marker='x')
# axes[1].legend(loc="best")
# plt.show()

#%% ============================ ROC curve section =====================================
# calculate the fpr and tpr for all thresholds of the classification

fpr_pcalog, tpr_pcalog, threshold = metrics.roc_curve(y_test, logreg.predict_proba(x_test_pcalog)[:,1])
fpr_log, tpr_log, threshold = metrics.roc_curve(y_test, logreg2.predict_proba(x_test_scaled)[:,1])
fpr_rf, tpr_rf, threshold = metrics.roc_curve(y_test, rnd_clf.predict_proba(x_test)[:,1])
fpr_knn, tpr_knn, threshold = metrics.roc_curve(y_test, knn.predict_proba(x_test)[:,1])
fpr_xgb, tpr_xgb, threshold = metrics.roc_curve(y_test, xgmodel.predict_proba(x_test)[:,1])
# %%
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_pcalog, tpr_pcalog, label = 'PCA Logistic')
plt.plot(fpr_log, tpr_log, label = 'Logistic only')
plt.plot(fpr_rf, tpr_rf, label = 'Random Forest')
plt.plot(fpr_knn, tpr_knn, label = 'knn')
plt.plot(fpr_xgb, tpr_xgb, label = 'XGBoost')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# %%
