import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#读取数据，以逗号隔开的csv文件
#选择文件
filename = input('输入训练数据文件')
dataset = pd.read_csv(filename, header=None, sep=',')
dataset = np.array(dataset)
X_len = len(dataset[0]) - 1
X = dataset[:,:X_len]
y = dataset[:,X_len:].reshape(-1,)
#标准化
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

#选择算法
clf_logreg = LogisticRegression()
clf_nvby = GaussianNB()
clf_knn = KNeighborsClassifier()
clf_tree = DecisionTreeClassifier()
clf_svm = SVC()
clf = clf_tree
clf.fit(X_train,y_train)
#expected = y
#predicted = clf.predict(X)
#print(metrics.classification_report(expected, predicted))
score = clf.score(X_test, y_test)
print(r'准确率:',score)
#print(predicted)
testname = input('输入预测文件')
preset = pd.read_csv(testname, header=None, sep=',')
x = np.array(preset)
predicted = clf.predict(x)
x = pd.DataFrame(x)
predicted = pd.DataFrame(predicted)
res = pd.concat([x, predicted], axis=1)
res.to_csv('res.csv',index=False,header=None)