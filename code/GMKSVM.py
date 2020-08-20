
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,matthews_corrcoef
from imblearn.metrics import specificity_score,sensitivity_score
from sklearn.model_selection import  KFold,StratifiedKFold
from sklearn.metrics.pairwise import linear_kernel,rbf_kernel

class GMKSVM(object):
    def __init__(self, C,gamma_list,feature_id,T = 10):
        self.C = C
        self.T = T
        self.gamma_list = gamma_list
        self.feature_id=feature_id

    def get_NLSV_sv(self, SV, label):
        list = []
        for ne_sv in SV:
            if label[ne_sv] == -1:
                list.append(ne_sv)
        return np.array(list)

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
            print(iterRecord)
            knernel_train = self.get_weight_kernel(np.array(xTrain), np.array(xTrain), self.feature_id,self.gamma_list)
            # print(knernel_train.shape)
            iterRecord=iterRecord+1
            svc = SVC(C = self.C, kernel='precomputed',class_weight='balanced',
                  degree=3)
            svc.fit(knernel_train, yTrain)
            sv = svc.support_  # This is support vector
            ne_sv = self.get_NLSV_sv(sv, yTrain)
            
            xTrain, yTrain, xNLSV, lastMar = self.rebuild(xTrain, yTrain, sv, xNLSV)  # rebuild sample
            if lastMar < 1* len(xPos):
                break

        for i in xNLSV:
            xlastTrain.append(i)
            ylastTrain.append(-1)
        


        Kernel_last_train=self.get_weight_kernel(np.array(xlastTrain),np.array(xlastTrain),self.feature_id,self.gamma_list)
     
        Kernel_last_test=self.get_weight_kernel(np.array(x_test),np.array(xlastTrain),self.feature_id,self.gamma_list)
       
        svc = SVC(C = self.C, kernel='precomputed',class_weight='balanced',
                  degree=3)
        svc.fit(Kernel_last_train,ylastTrain)
        y_pre=svc.predict(Kernel_last_test)
        ACC=accuracy_score(y_test,y_pre)
        SN=sensitivity_score(y_test,y_pre)
        SP=specificity_score(y_test,y_pre)
        MCC=matthews_corrcoef(y_test,y_pre)
        return SN,SP,ACC,MCC

    
    def get_weight_kernel(self,X,Y,feature_id,gamma_list):
        m = feature_id.shape[0]
        num_1 = X.shape[0]
        num_2 = Y.shape[0]
        Kernel = np.zeros((num_1, num_2))
        for i in range(0, m):
            kk_kernel = rbf_kernel(X[:, feature_id[i, 0]:feature_id[i, 1]], Y[:, feature_id[i, 0]:feature_id[i, 1]] ,gamma_list[i])
            Kernel=Kernel+kk_kernel
        
        Kernel = Kernel / m
        return Kernel

    # def get_weight_kernel(self,X,Y,feature_id,gamma_list):
    #     m = feature_id.shape[0]
    #     num_1 = X.shape[0]
    #     num_2 = Y.shape[0]
    #     Kernel = np.zeros((num_1, num_2))
    #     for i in range(0, m):
    #         kk_kernel = linear_kernel(X[:, feature_id[i, 0]:feature_id[i, 1]], Y[:, feature_id[i, 0]:feature_id[i, 1]])
    #         Kernel=Kernel+kk_kernel
        
    #     Kernel = Kernel / m
    #     return Kernel

def get_Kfold(feature,label,index):
    feature_new=[]
    label_new = []
    for i in range(0,len(index)):
        feature_new.append(feature[index[i]])
        label_new.append(label[index[i]])
        # print(index[i])
    return feature_new,label_new




if __name__ == '__main__':
    feature=np.loadtxt('/home/yangchao/YC2/RBP129/feature+label/feature_normalize.csv',delimiter=',',encoding='utf-8').tolist()
    label=np.loadtxt('/home/yangchao/YC2/RBP129/feature+label/label.csv',delimiter=',',encoding='utf-8').tolist()
    feature_id = np.array([[0, 20], [20, 40],[40,55],[55,76],[76,416]])
    gamma_list=np.array([0.03125,0.5,1,0.25,0.125])
    C=np.logspace(-5,5,base=2,num=11)
    for i in range(0,11):
        c=C[i];
        RBP=GMKSVM(c,gamma_list,feature_id)
        kf=StratifiedKFold(n_splits=5)
        for train_index,test_index in kf.split(feature,label):
            # print(train_index,test_index)
            train_feature,train_label=get_Kfold(feature,label,train_index)
            test_feature, test_label = get_Kfold(feature, label, test_index)
            SN, SP, ACC, MCC=RBP.fit(train_feature,train_label,test_feature,test_label)



