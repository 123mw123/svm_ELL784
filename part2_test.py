import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import csv
import matplotlib.pyplot as plt
import pylab as pl


data = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')
data2 = pd.read_csv('2015EE10466.csv')

features_test1 = np.array(data2.iloc[:,:25])
labels_test1 = np.array(data2.iloc[:, 25:])
features_train = np.array(data.iloc[:,:25])
labels_train = np.array(data.iloc[:, 25:])
features_test = np.array(test.iloc[:,:])

kf = StratifiedKFold(n_splits=3)
C_range = 2. ** np.arange(-5, 15)
gamma_range = 2. ** np.arange(-15, 3)
degree_range = np.arange(1,20)
kernel = ['linear']

param_grid = dict(gamma=[6.103515625e-05], C=[0.0625],max_iter= [800])

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=3))
t0= time()
grid.fit(features_train,labels_train.ravel())
pred = grid.predict(features_test)
print pred
print 'fitting time',time()-t0
print grid.best_estimator_
print grid.score(features_train,labels_train.ravel())
print grid.score(features_test1,labels_test1.ravel()),'actual'

with open('eggs.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in pred:

        spamwriter.writerow([i])
    spamwriter.writerow(["endlast"])

for i in pred:
    print i

