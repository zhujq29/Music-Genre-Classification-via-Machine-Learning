#!/usr/bin/env python
# coding: utf-8

 


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

music = pd.read_csv('train.csv')

music.isnull().sum()


music = music.dropna()
music.shape

music.describe(include = 'all')


music.groupby('Class').size()


l = list(range(0,11))
labels = ['Hiphop', 'Instrumental', 'Rock']
music1 = music[music['Class'].isin(l)]

fig, ax = plt.subplots(ncols = 3,nrows = 3, figsize=(20, 15))
g = sns.boxplot(data = music1, x = 'Class', y='Popularity', ax = ax[0,0])
g.set_title('Popularity')
#g.set_xticklabels(labels)
f = sns.boxplot(data = music1, x = 'Class', y='danceability', ax = ax[0,1])
f.set_title('danceability')
#f.set_xticklabels(labels)
g = sns.boxplot(data = music1, x = 'Class', y='loudness', ax = ax[0,2])
g.set_title('loudness')
#g.set_xticklabels(labels)
f = sns.boxplot(data = music1, x = 'Class', y='acousticness', ax = ax[1,0])
f.set_title('acousticness')
#f.set_xticklabels(labels)
g = sns.boxplot(data = music1, x = 'Class', y='speechiness', ax = ax[1,1])
g.set_title('speechiness')
#g.set_xticklabels(labels)
f = sns.boxplot(data = music1, x = 'Class', y='instrumentalness', ax = ax[1,2])
f.set_title('instrumentalness')
#f.set_xticklabels(labels)
g = sns.boxplot(data = music1, x = 'Class', y='liveness', ax = ax[2,0])
g.set_title('liveness')
#g.set_xticklabels(labels)
f = sns.boxplot(data = music1, x = 'Class', y='valence', ax = ax[2,1])
f.set_title('valence')
#f.set_xticklabels(labels)
f = sns.boxplot(data = music1, x = 'Class', y='tempo', ax = ax[2,2])
f.set_title('tempo')
#f.set_xticklabels(labels)
plt.show()


l = [5, 7, 10]
labels = ['Hiphop', 'Instrumental', 'Rock']
music1 = music[music['Class'].isin(l)]

fig, ax = plt.subplots(ncols = 3,nrows = 2, figsize=(20, 15))
f = sns.boxplot(data = music1, x = 'Class', y='duration_in min/ms', ax = ax[0,0])
f.set_title('duration_in min/ms')
f.set_xticklabels(labels)

g = sns.boxplot(data = music1, x = 'Class', y='loudness', ax = ax[0,1])
g.set_title('loudness')
g.set_xticklabels(labels)

f = sns.boxplot(data = music1, x = 'Class', y='acousticness', ax = ax[1,0])
f.set_title('acousticness')
f.set_xticklabels(labels)

g = sns.boxplot(data = music1, x = 'Class', y='speechiness', ax = ax[1,1])
g.set_title('speechiness')
g.set_xticklabels(labels)

f = sns.boxplot(data = music1, x = 'Class', y='instrumentalness', ax = ax[1,2])
f.set_title('instrumentalness')
f.set_xticklabels(labels)

f = sns.boxplot(data = music1, x = 'Class', y='valence', ax = ax[0,2])
f.set_title('valence')
f.set_xticklabels(labels)

plt.show()


music.loc[music.Class == 10, 'class'] = 0 # Rock Music
music.loc[music.Class == 7, 'class'] = 1 # Instrumental Music Genre
music.loc[music.Class == 5, 'class'] = 2 # HipHop Music Genre
# music.loc[music.Class == 9, 'class'] = 1 # Pop
# music.loc[music.Class == 8, 'class'] = 2 # Metal Music Genre
# music1_scale.loc[music1_scale.Class == 9, 'class'] = 3
# music1_scale.loc[music1_scale.Class == 1, 'class'] = 4
# music1_scale['class'].fillna(3,inplace = True)


music = music.dropna()
music = music.drop('Class', axis=1)


music.describe()


music.groupby('class').size()


music_cate = music[['key', 'mode', 'time_signature', 'class']]
music_cate.head()

music1 = music.drop(columns=['Artist Name','Track Name','mode','class','time_signature','key'])
music1.head()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
music1_scale = pd.DataFrame(sc.fit_transform(music1), columns=music1.columns)

music1_corr = music1_scale.corr()


f, ax = plt.subplots(figsize=(25,20))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(music1_corr, cmap=cmap, vmax=.3, center=0, annot=True,square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.show()

music1_scale.head()


music1_scale['class'] = music['class'].tolist() 
music1_scale['key'] = music['key'].tolist() 
music1_scale['mode'] = music['mode'].tolist() 
music1_scale['time_signature'] = music['time_signature'].tolist()
music1_scale 

fig, ax = plt.subplots(ncols = 1,nrows = 3, figsize=(5, 15))
g = sns.boxplot(data = music1_scale, x = 'class', y='Popularity', ax = ax[0])
g.set_title('Popularity')
#f = sns.boxplot(data = music1_scale, x = 'class', y='danceability', ax = ax[0,1])
#f.set_title('danceability')
g = sns.boxplot(data = music1_scale, x = 'class', y='loudness', ax = ax[1])
g.set_title('loudness')
f = sns.boxplot(data = music1_scale, x = 'class', y='acousticness', ax = ax[2])
f.set_title('acousticness')
plt.show()


fig, ax = plt.subplots(ncols = 2,nrows = 3, figsize=(15, 10))
g = sns.histplot(data = music1_scale, x = 'Popularity',kde=True,ax = ax[0,0])
g.set(xlim=(-3,3.5))
sns.histplot(data = music1_scale, x = 'danceability',kde=True,ax = ax[0,1])
sns.histplot(data = music1_scale, x = 'energy',kde=True,ax = ax[1,0])
sns.histplot(data = music1_scale, x = 'loudness',kde=True,ax = ax[1,1])
sns.histplot(data = music1_scale, x = 'acousticness',kde=True,ax = ax[2,0])
sns.histplot(data = music1_scale, x = 'instrumentalness',kde=True,ax = ax[2,1])
plt.show()


# - logistic regression
# - KNN
# - decision tree
# - rf 
# - NN
# - svm
# - pca
# - sdm
# - hierarchical clustering



from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(music1_scale, music1_scale["class"]):
    strat_train_set = music1_scale.iloc[train_index]
    strat_test_set = music1_scale.iloc[test_index]



strat_train_set = strat_train_set.reset_index(drop = True)
strat_test_set = strat_test_set.reset_index(drop = True)



train_X = strat_train_set.drop('class', axis = 1)
test_X = strat_test_set.drop('class', axis = 1)
train_y = strat_train_set['class']
test_y = strat_test_set['class']



train_y.unique()


from sklearn.metrics import classification_report, confusion_matrix


# ### KNN


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(train_X, train_y)
pred_y = knn_model.predict(test_X)
train_knn = knn_model.score(train_X, train_y)
test_knn = knn_model.score(test_X, test_y)


from sklearn.model_selection import GridSearchCV

grid_clf_knn = GridSearchCV(cv=10, estimator=KNeighborsClassifier(), n_jobs=-1,
             param_grid={'n_neighbors': [1, 3, 5, 10, 20]})
grid_clf_knn.fit(train_X, train_y)



grid_clf_knn.best_estimator_



y_pred = grid_clf_knn.predict(test_X)

train_knn = grid_clf_knn.score(train_X, train_y)
test_knn = grid_clf_knn.score(test_X, test_y)



grid_clf_knn.best_estimator_



m_knn = confusion_matrix(test_y, y_pred)
knn_byclass = m_knn.diagonal()/m_knn.sum(axis=1)
knn_byclass = knn_byclass.tolist()
knn_byclass



knn_byclass[1]


# ### Decision tree

 


from sklearn.tree import DecisionTreeClassifier

tr_model = DecisionTreeClassifier(
                random_state=42, 
                criterion='entropy',
                splitter='best', 
                max_depth=5, 
                min_samples_split=2)

tr_model.fit(train_X, train_y)
y_pred = tr_model.predict(test_X)

train_tr = tr_model.score(train_X, train_y)
test_tr = tr_model.score(test_X, test_y)


 


train_tr


 


test_tr


 


from sklearn.model_selection import GridSearchCV

param_grid = {
                'min_samples_split': [2, 3, 4, 5, 6, 7, 8], 
                'max_depth': [2, 5, 7, 9],
             }

grid_clf_tr = GridSearchCV(tr_model, param_grid, cv=10, n_jobs=-1, refit = True)
grid_clf_tr.fit(train_X, train_y)

y_pred = grid_clf_tr.predict(test_X)

train_tr = grid_clf_tr.score(train_X, train_y)
test_tr = grid_clf_tr.score(test_X, test_y)


 


grid_clf_tr.best_estimator_


 


train_tr


 


test_tr 


 


m_tr = confusion_matrix(test_y, y_pred)
tr_byclass = m_tr.diagonal()/m_tr.sum(axis=1)
tr_byclass = tr_byclass.tolist()
tr_byclass


# ### Random forest

 


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42, 
    criterion='entropy',
    max_depth=8, 
    min_samples_split=2)

rf_model.fit(train_X, train_y)
y_pred = rf_model.predict(test_X)

train_rf = rf_model.score(train_X, train_y)
test_rf = rf_model.score(test_X, test_y)


 


train_rf


 


test_rf


 


from sklearn.model_selection import GridSearchCV

param_grid = {
                'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 50, 100],
                'max_depth': [2, 3, 4, 5, 7, 8],
             }

grid_clf_rf = GridSearchCV(rf_model, param_grid, cv=10, n_jobs=-1, refit = True)
grid_clf_rf.fit(train_X, train_y)

y_pred = grid_clf_rf.predict(test_X)

train_rf = grid_clf_rf.score(train_X, train_y)
test_rf = grid_clf_rf.score(test_X, test_y)


 


grid_clf_rf.best_estimator_




m_rf = confusion_matrix(test_y, y_pred)
rf_byclass = m_rf.diagonal()/m_rf.sum(axis=1)
rf_byclass = rf_byclass.tolist()
rf_byclass


 


model = RandomForestClassifier(
    criterion='entropy', max_depth=8, n_estimators=25,
    random_state=42)

model.fit(train_X,train_y)
y_pred = model.predict(test_X)

feat_importance = model.feature_importances_
pd.DataFrame({'Feature Importance':feat_importance},
            index=list(train_X)).sort_values('Feature Importance').plot(kind='barh')


# ### Logistic regression

 


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(solver='liblinear',multi_class="auto")
log_model.fit(train_X, train_y)
y_pred = log_model.predict(test_X)

train_log = log_model.score(train_X, train_y)
test_log = log_model.score(test_X, test_y)


 


m_log = confusion_matrix(test_y, y_pred)
log_byclass = m_log.diagonal()/m_log.sum(axis=1)
log_byclass = log_byclass.tolist()
log_byclass


# ### AdaBoost

 


from sklearn.ensemble import AdaBoostClassifier

adb_model = AdaBoostClassifier(
    n_estimators=150,
    learning_rate=0.4,
    random_state=42)

adb_model.fit(train_X, train_y)
y_pred = adb_model.predict(test_X)

train_adb = adb_model.score(train_X, train_y)
test_adb = adb_model.score(test_X, test_y)



from sklearn.model_selection import GridSearchCV
param_grid = {
                 'n_estimators': [10, 15, 20, 25, 30, 35, 40, 50, 100, 150],
                 'learning_rate': [0.2, 0.4, 0.5, 0.6, 0.8, 1],
             }

grid_clf_adb = GridSearchCV(adb_model, param_grid, cv=10, n_jobs=-1, refit = True)
grid_clf_adb.fit(train_X, train_y)

y_pred = grid_clf_adb.predict(test_X)

train_adb = grid_clf_adb.score(train_X, train_y)
test_adb = grid_clf_adb.score(test_X, test_y)


 


grid_clf_adb.best_estimator_




m_ada = confusion_matrix(test_y, y_pred)
ada_byclass = m_ada.diagonal()/m_ada.sum(axis=1)
ada_byclass = ada_byclass.tolist()
ada_byclass


 


model = AdaBoostClassifier(learning_rate=0.8, n_estimators=25, random_state=42)

model.fit(train_X,train_y)
y_pred = model.predict(test_X)

feat_importance = model.feature_importances_
pd.DataFrame({'Feature Importance':feat_importance},
            index=list(train_X)).sort_values('Feature Importance').plot(kind='barh')


 


from sklearn.ensemble import VotingClassifier


clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf2 = RandomForestClassifier(criterion='entropy', max_depth=8, n_estimators=25,
                       random_state=42)
clf3 = AdaBoostClassifier(learning_rate=0.8, n_estimators=25, random_state=42)


# ### SVM

 


from sklearn import svm
svm_model = svm.SVC(gamma="scale",kernel="rbf")
svm_model.fit(train_X, train_y)
y_pred = svm_model.predict(test_X)

train_svm = svm_model.score(train_X, train_y)
test_svm = svm_model.score(test_X, test_y)



m_svm = confusion_matrix(test_y, y_pred)
svm_byclass = m_svm.diagonal()/m_svm.sum(axis=1)
svm_byclass = svm_byclass.tolist()
svm_byclass


# ### MLP

from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(activation='tanh', hidden_layer_sizes=(20,), random_state=42)
mlp_model.fit(train_X, train_y)
y_pred = mlp_model.predict(test_X)

train_mlp = mlp_model.score(train_X, train_y)
test_mlp = mlp_model.score(test_X, test_y)


from sklearn.model_selection import GridSearchCV

param_grid = {
                'hidden_layer_sizes': [(10,30,10),(20,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant','adaptive'],
             }

grid_clf_mlp = GridSearchCV(mlp_model, param_grid, cv=10, n_jobs=-1, refit = True)
grid_clf_mlp.fit(train_X, train_y)

y_pred = grid_clf_mlp.predict(test_X)

train_mlp = grid_clf_mlp.score(train_X, train_y)
test_mlp = grid_clf_mlp.score(test_X, test_y)


 


grid_clf_mlp.best_estimator_
 


m_mlp = confusion_matrix(test_y, y_pred)
mlp_byclass = m_mlp.diagonal()/m_mlp.sum(axis=1)
mlp_byclass = mlp_byclass.tolist()
mlp_byclass



methods = ['KNN', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'AdaBoost', 'SVM', 'MLP']
train_accuracy = [train_knn, train_tr, train_rf, train_log, train_adb, train_svm, train_mlp]
test_accuracy = [test_knn, test_tr, test_rf, test_log, test_adb, test_svm, test_mlp]
rock_accuracy = [knn_byclass[0], tr_byclass[0], rf_byclass[0], log_byclass[0], ada_byclass[0], svm_byclass[0], mlp_byclass[0]]
instrumental_accuracy = [knn_byclass[1], tr_byclass[1], rf_byclass[1], log_byclass[1], ada_byclass[1], svm_byclass[1], mlp_byclass[1]]
hiphop_accuracy = [knn_byclass[2], tr_byclass[2], rf_byclass[2], log_byclass[2], ada_byclass[2], svm_byclass[2], mlp_byclass[2]]


zipped = list(zip(methods, test_accuracy, train_accuracy, rock_accuracy, instrumental_accuracy, hiphop_accuracy))
result = pd.DataFrame(zipped, columns=['Methods', 'Overall Training', 'Overall Testing', 'Rock', 'Instrumental', 'Hip Hop'])



result.round(4).sort_values('Overall Testing', ascending=False)