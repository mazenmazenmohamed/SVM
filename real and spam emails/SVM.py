from scipy.io import loadmat
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import svm 

#functions
def plot_data(features,classify,size):
    pos = (classify==1).ravel()
    neg = (classify==0).ravel()
    plt.scatter(features[pos,0],features[pos,1],s=size,c='b',marker='+',linewidth=1)
    plt.scatter(features[neg,0],features[neg,1],s=size,c='b',marker='o',linewidth=1)

#example 1 ........

# data1=loadmat('C:\\1.PC\\Career\\ML\\cods\\SVM\\real and spam emails\\ex6data1')
# features1=data1['X']
# classify1=data1['y']
# print(features1)
# print(classify1)
# plot_data(features1,classify1,100)
# clf = svm.SVC(C=100.0,kernel="linear")
# clf.fit(features1,classify1)

# w = clf.coef_[0]
# print(w)
# a = -w[0] / w[1]

# xx = np.linspace(0,12)
# yy = a * xx - clf.intercept_[0] / w[1]

# h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

# plt.scatter(features1[:, 0], features1[:, 1], c = classify1)
# plt.legend()
# plt.show()

#example 2 : Non linear SVM........

# data2=loadmat('C:\\1.PC\\Career\\ML\\cods\\SVM\\real and spam emails\\ex6data2')
# features2=data2['X']
# classify2=data2['y']
# print(features2)
# print(classify2)
# plot_data(features2,classify2,10)

# clf = svm.SVC(C=1.0,kernel="rbf",gamma=6)
# clf.fit(features2,classify2)



# plt.scatter(features2[:, 0], features2[:, 1], c = classify2)
# plt.legend()
# plt.show()

#training ........
data_train =loadmat('C:\\1.PC\\Career\\ML\\cods\\SVM\\real and spam emails\\spamTrain')
data_test =loadmat('C:\\1.PC\\Career\\ML\\cods\\SVM\\real and spam emails\\spamTest')

features_train =data_train['X']
features_test =data_test['Xtest']

classify_train =data_train['y'].ravel()
classify_test =data_test['ytest'].ravel()

print(features_train.shape,classify_train.shape,features_test.shape,classify_test.shape)

svc = svm.SVC() 
svc.fit(features_train,classify_train)
#testing ............ 
#print('Test Accuracy = {0}%'.format(np.round(svc.score(features_test, classify_test)*100,2)))

print(features_test)
print(features_test[0])
print(svc.predict(np.array(features_test[0]).reshape(1,-1)))



















