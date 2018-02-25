from tkinter import *
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
import tkinter.messagebox as messagebox
from tkinter.filedialog import askdirectory
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()
            

    #页面布局
    def createWidgets(self):
        self.fileLabel = Label(self, text='文件地址：')
        self.fileLabel.grid(row=0, sticky=W)
        self.filePath = Entry(self, textvariable = path)
        self.filePath.grid(row=0, column=1, sticky=W)
        self.fileButton = Button(self, text='提交', command=self.choic)
        self.fileButton.grid(row=0, column=2, sticky=W)
        #self.selectButton = Button(self, text='选择文件', command=self.selectPath)
        #self.selectButton.grid(row=0, column=2, sticky=W)
        self.fileLabel = Label(self, text='选择分类算法：')
        self.fileLabel.grid(row=1, sticky=W)
        self.var = IntVar()
        self.reg = Radiobutton(self, text='逻辑回归',variable = self.var, value='1')
        self.reg.grid(row=1, column=1)
        self.nvby = Radiobutton(self, text='朴素贝叶斯',variable = self.var, value='2')
        self.nvby.grid(row=1, column=2, sticky=W)
        self.knn = Radiobutton(self, text='K近邻', variable = self.var, value='3')
        self.knn.grid(row=1, column=3, sticky=W)
        self.tree = Radiobutton(self, text='决策树', variable = self.var, value='4')
        self.tree.grid(row=1, column=4, sticky=W)
        self.svm = Radiobutton(self, text='支持向量机', variable = self.var, value='5')
        self.svm.grid(row=1, column=5, sticky=W)
        self.alertButton = Button(self, text='提交', command=self.learn)
        self.alertButton.grid(row=3, column=1)
        self.predictLabel = Label(self, text='预测文件地址：')
        self.predictLabel.grid(row=6, sticky=W)
        self.predictPath = Entry(self, textvariable = path)
        self.predictPath.grid(row=6,column=1,sticky=W) 
        self.predictButton = Button(self, text='提交', command=self.pred)
        self.predictButton.grid(row=7,column=1)
    
    def selectPath(self):
        #路径选择
        path.set(askdirectory())
    
    def choic(self):
        value = self.var.get() or '2'
        #if value == 1:
            #clf = LogisticRegression()
        #elif value == 2:
            #clf =  GaussianNB()
        #elif value == 3:
            #clf = KNeighborsClassifier()
        #elif value == 4:
            #clf = DecisionTreeClassifier()
        elif value == 5:
            self.fiveLabel = Label(self, text='选择核函数与罚数')
            self.fiveLabel.grid(row=4, sticky=W)
            self.fiveText = Entry(self)
            self.fiveText.grid(row=4, column=1, sticky=W)
        
    def learn(self):
        
        #选择分类算法
        value = self.var.get() or '2'
        if value == 1:
            clf = LogisticRegression()
        elif value == 2:
            clf =  GaussianNB()
        elif value == 3:
            clf = KNeighborsClassifier()
        elif value == 4:
            clf = DecisionTreeClassifier()
        elif value == 5:
            clf = SVC()
        #数据预处理
        filename = self.filePath.get()
        dataset = pd.read_csv(filename, header=None, sep=',')
        dataset = np.array(dataset)
        X_len = len(dataset[0]) - 1
        X = dataset[:,:X_len]
        y = dataset[:,X_len:].reshape(-1,)
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)
        #训练
        clf.fit(X_train,y_train)
        joblib.dump(clf, 'clf.model')
        #score = cross_val_score(clf, X_test, y_test, cv = 10, scoring='accuracy', n_jobs=-1).mean()
        score = clf.score(X_test, y_test)
        messagebox.showinfo('准确率','准确率： %s' % score)
        
        
    def pred(self):
        testname = self.predictPath.get()
        preset = pd.read_csv(testname, header=None, sep=',')
        x = np.array(preset)
        predicted = clf.predict(x)
        x = pd.DataFrame(x)
        predicted = pd.DataFrame(predicted)
        res = pd.concat([x, predicted], axis=1)
        res.to_csv('res.csv',index=False,header=None)
        messagebox.showinfo('完成','预测完成，已保存到res.csv')
app = Application()
# 设置窗口标题:
app.master.title('分类器')
# 主消息循环:
app.mainloop()