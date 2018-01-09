import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation

#读取数据，以逗号隔开的csv文件
dataset = pd.read_csv('dataset.csv', header=None, sep=',')
dataset = np.array(dataset)
X_len = len(dataset[0]) - 1
X = dataset[:,:X_len]
y = dataset[:,X_len:].reshape(-1,)
#标准化
#normalized_X = preprocessing.normalize(X)
#数据分割
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)
#选择算法
clf = SCV()
#训练
clf.fit(X_train,y_train)
expected = y
predicted = clf.predict(X)
print(metrics.classification_report(expected, predicted))
clf.score(X_test, y_test)