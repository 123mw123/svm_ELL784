import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import pylab as pl
from time import time


data = pd.read_csv('2015EE10466.csv')

data = data.reindex(np.random.permutation(data.index))
#print data2

train, test = np.split(data.sample(frac=1), [int(.8 * len(data))])

features_train = np.array(train.iloc[:,:10])
labels_train = np.array(train.iloc[:, 25:])

features_test = np.array(test.iloc[:, :10])
labels_test = np.array(test.iloc[:,25:])
kf = StratifiedKFold(n_splits=3)
C_range = 2. ** np.arange(-8, 8)
gamma_range = 2. ** np.arange(-25, -15)
degree_range = np.arange(1,20)

coef0_range =np.arange(-2,2)

i=0

train_error = []
val_error = []
for c in C_range:
    clf = SVC(C=c, kernel='linear',max_iter=2000)
    train_score =0
    val_score=0
    for train_index, val_index in kf.split(features_train, labels_train.ravel()):
        X_train, X_val = features_train[train_index], features_train[val_index]
        Y_train, Y_val = labels_train[train_index], labels_train[val_index]

        clf.fit(X_train, Y_train.ravel())
        train_score = train_score + clf.score(X_train, Y_train.ravel())
        # print log_loss(Y_train.ravel(),clf.predict(X_train)),'train'
        val_score = val_score + clf.score(X_val, Y_val.ravel())

    i = i + 1
    print i

    train_error = train_error + [1 - (train_score / 4)]
    val_error = val_error + [1 - (val_score / 4)]

print train_error,'linear'
print val_error,'linear'


plt.plot(C_range,train_error,label='linear')
plt.plot(C_range,val_error)
plt.show()

###RBF kernel

param_grid = dict(gamma=gamma_range, C=C_range,max_iter= [10])

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=3))
t0= time()
grid.fit(features_train,labels_train.ravel())
print "fitting time",time()-t0
print grid.best_estimator_
print grid.score(features_train,labels_train.ravel()),'train score'
print grid.score(features_test,labels_test.ravel()),'test score'

score_dict = grid.grid_scores_

# We extract just the scores
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(scores, interpolation='nearest', cmap='jet')
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
pl.yticks(np.arange(len(C_range)), C_range)
pl.show()




### Polynomial kernal

param_grid = dict(kernel =['poly'],gamma=gamma_range,coef0 =coef0_range, degree=degree_range, C=C_range,max_iter= [10])

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=3))
t0 =time()
grid.fit(features_train,labels_train.ravel())

print '###PolyKernal###'
print "fitting time",time()-t0


score_dict = grid.grid_scores_

# We extract just the scores
print grid.best_estimator_
print grid.score(features_test,labels_test.ravel()),"score"
for x in score_dict:
    print x

print grid.best_estimator_
print grid.score(features_train,labels_train.ravel()),'train score'
print grid.score(features_test,labels_test.ravel()),'test score'
