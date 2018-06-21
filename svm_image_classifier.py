import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import pylab as pl

data = pd.read_csv('2015EE10466.csv')


image1 = data.iloc[0:300,:10]
labels1 = data.iloc[0:300,25:]

data1= pd.concat([image1,labels1],axis=1)

data1 = data1.reindex(np.random.permutation(data1.index))
#print data1

image2 = data.iloc[300:600,:10]
labels2 = data.iloc[300:600,25:]
data2= pd.concat([image2,labels2],axis=1)

data2 = data2.reindex(np.random.permutation(data2.index))
#print data2

train1, test1 = np.split(data1.sample(frac=1), [int(.8 * len(data1))])
train2, test2 = np.split(data2.sample(frac=1), [int(.8 * len(data2))])

train = pd.concat([train1,train2])
test = pd.concat([test1,test2])

features_train = np.array(train.iloc[:, :10])
labels_train = np.array(train.iloc[:,10:])

features_test = np.array(test.iloc[:, :10])
labels_test = np.array(test.iloc[:,10:])



print features_train.shape,labels_train.shape

kf = StratifiedKFold(n_splits=3)

C_range = 10. ** np.arange(-1, 3)
gamma_range = 10. ** np.arange(-8, -3)
degree_range = np.arange(1,4)
coef0_range =np.arange(-17,-14)
# tuning C and degree
param_grid = dict(kernel = ['poly'],degree=degree_range, C=C_range,max_iter= [100],gamma= gamma_range,coef0= coef0_range)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=3))
grid.fit(features_train,labels_train.ravel())

print '###PolyKernal###'


score_dict = grid.grid_scores_
#print score_dict

for x in score_dict:
    print x
print grid.best_estimator_
print grid.score(features_train,labels_train),'train score'
print grid.score(features_test,labels_test.ravel()),"test score"

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------##
kernel = ['linear']
l = 0
while l<len(kernel):
    train_error = []
    val_error = []

    for c in C_range:
        clf = SVC(C=c, kernel=kernel[l],max_iter=1000)
        train_score =0
        val_score=0

        for train_index, val_index in kf.split(features_train,labels_train.ravel()):
            X_train, X_val = features_train[train_index], features_train[val_index]
            Y_train, Y_val = labels_train[train_index], labels_train[val_index]
            clf.fit(X_train,Y_train.ravel())

            train_score = train_score+ clf.score(X_train,Y_train.ravel())
            #print log_loss(Y_train.ravel(),clf.predict(X_train)),'train'
            val_score = val_score + clf.score(X_val,Y_val.ravel())

        train_error = train_error +[1-(train_score/4)]
        val_error = val_error + [1-(val_score/4)]

    print train_error,kernel[l]
    print val_error,kernel[l]
    print 'test_accuracy', clf.score(features_test,labels_test.ravel())
    plt.plot(C_range,train_error,label=kernel[l])
    plt.plot(C_range,val_error)
    plt.show()
    l=l+1



param_grid = dict(gamma=gamma_range, C=C_range, max_iter= [1000])

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=3))
grid.fit(features_train,labels_train.ravel())

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






