
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,matthews_corrcoef
from imblearn.metrics import specificity_score,sensitivity_score
from sklearn.model_selection import  KFold,StratifiedKFold

class GSVM(object):
    def __init__(self, C,gamma,
                 T = 10,class_weight = 'balanced',
                 degree = 3, kernel='rbf'
                 ):
        self.C = C
        self.T = T
        self.class_weight = class_weight
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self.allSVC = SVC(C = self.C, class_weight=self.class_weight,
                      degree=self.degree, gamma = self.gamma,
                      kernel= self.kernel)

    def rebuild(self, xTrain, yTrain, sv, xNLSV):  # rebuild SVC
        xNew = []
        yNew = []
        count = 0
        for i in range(0, len(yTrain)):
            if yTrain[i] == 1:
                xNew.append(xTrain[i])
                yNew.append(yTrain[i])
            else:
                if i not in sv:
                    xNew.append(xTrain[i])
                    yNew.append(yTrain[i])
                    count += 1
                else:
                    xNLSV.append(xTrain[i])
        return xNew, yNew, xNLSV, count

    def fit(self, x, y,x_test,y_test):
        #
        xPos = []
        xNeg = []
        xTrain = []
        yTrain = []
        xlastTrain = []
        ylastTrain = []

        for i in range(0, len(y)):
            if y[i] == 1:
                xPos.append(x[i])
                xlastTrain.append(x[i])
                ylastTrain.append(y[i])
            else:
                xNeg.append(x[i])
            xTrain.append(x[i])
            yTrain.append(y[i])
        xNLSV = []
        iterRecord = 0
        for i in range(0, self.T):
            svc = SVC(C = self.C, class_weight=self.class_weight,
                      degree=self.degree, gamma = self.gamma,
                      kernel= 'linear')
            print (iterRecord)
            iterRecord += 1
            svc.fit(xTrain, yTrain)
            sv = svc.support_  # This is support vector
            xTrain, yTrain, xNLSV, lastMar = self.rebuild(xTrain, yTrain, sv, xNLSV)  # rebuild sample
            #print (lastMar)
            if lastMar < 0.1 * len(xPos):
                break

        for i in xNLSV:
            xlastTrain.append(i)
            ylastTrain.append(-1)

        self.allSVC.fit(xlastTrain, ylastTrain)

        y_pre=self.allSVC.predict(x_test)
        ACC=accuracy_score(y_test,y_pre)
        SN=sensitivity_score(y_test,y_pre)
        SP=specificity_score(y_test,y_pre)
        MCC=matthews_corrcoef(y_test,y_pre)
        return SN,SP,ACC,MCC

def get_Kfold(feature,label,index):
    feature_new=[]
    label_new = []
    for i in range(0,len(index)):
        feature_new.append(feature[index[i]])
        label_new.append(label[index[i]])
        # print(index[i])
    return feature_new,label_new


if __name__ == '__main__':
    train_feature=np.loadtxt('/home/yangchao/YC2/RBP60/train_data/PSSM+Fre.csv',delimiter=',',encoding='utf-8').tolist()
    train_label=np.loadtxt('/home/yangchao/YC2/RBP60/train_data/label.csv',delimiter=',',encoding='utf-8').tolist()
    test_feature=np.loadtxt('/home/yangchao/YC2/RBP60/test_data/PSSM+Fre.csv',delimiter=',',encoding='utf-8').tolist()
    test_label=np.loadtxt('/home/yangchao/YC2/RBP60/test_data/label.csv',delimiter=',',encoding='utf-8').tolist()
    C=np.logspace(-5,1,base=2,num=7)
    Gamma=np.logspace(-5,-2,base=2,num=4)
    for i in range(0,7):
        for j in range(0,4):
            c=C[i]
            gamma=Gamma[j]
            RBP=GSVM(c,gamma)
            SN, SP, ACC, MCC=RBP.fit(train_feature,train_label,test_feature,test_label)
            print(SN,SP,ACC,MCC)
            str1=str(SN)+','+str(SP)+','+str(ACC)+','+str(MCC)+','+str(c)+','+str(gamma)+'\n'
            fw=open('result.txt','a')
            fw.write(str1)
            fw.close()








