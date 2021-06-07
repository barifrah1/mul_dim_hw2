import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score

def norm(mat,d):
    for i in range(d):
        mat[i]=mat[i]/sum(mat[i])
    return mat

data =  pd.read_csv('arcene_train.data', sep=" ",header=None)
lable=pd.read_csv('arcene_train.labels',sep=" ",header=None)
lable=lable.values
data=data.drop(data.columns[10000], axis=1)
D=data.shape[1]
d_list=[500,600,700,800,900,1000]
result={}
for d in d_list:
    #d=900
    P=np.random.randn(D, d)
    P=norm(P,d)
    pro=data@P

    #dist = numpy.linalg.norm(a-b)
    data_val=data.values
    old_row_22=data_val[ 21 , : ]
    old_row_43=data_val[ 44 , : ]
    old_dist = np.linalg.norm(old_row_22-old_row_43)
    print('dist betwenn 22 to 43 is:',old_dist)


    new_row_22=P[ 21 , : ]
    new_row_43=P[ 44 , : ]
    new_dist = np.linalg.norm(new_row_22-new_row_43)

    print('new dist between 22 to 43 is:',new_dist)
    print('ratio between original and new is:',old_dist/new_dist)

    kf = KFold(n_splits=10,shuffle=False)
    lable
    accuracy_list_original=[]
    accuracy_list_new=[]
    for train_index, test_index in kf.split(data):
        # Split train-test
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = lable[train_index], lable[test_index]
        clf = svm.SVC()
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        accuracy=accuracy_score(pred,y_test)
        accuracy_list_original.append(accuracy)
        
    lable
    accuracy_list_new=[]
    for train_index, test_index in kf.split(pro):
        # Split train-test
        X_train, X_test = pro.iloc[train_index], pro.iloc[test_index]
        y_train, y_test = lable[train_index], lable[test_index]
        clf = svm.SVC()
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        accuracy=accuracy_score(pred,y_test)
        accuracy_list_new.append(accuracy)   
        
    print('mean correct classification rate for the original data', np.mean(accuracy_list_original))
    print('mean correct classification rate for the reduced data',np.mean(accuracy_list_new))
    result[d]=[np.mean(accuracy_list_original),np.mean(accuracy_list_new)]
print('results as function of d',result)  
    
