import function
import math
import csv
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
import datetime
import matplotlib.pyplot as plt
import pickle
import joblib

import pandas as pd
from sklearn import datasets
from scipy.stats import randint
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import HalvingRandomSearchCV
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from scipy.signal import savgol_filter


# filelist_test=["samples_N_test","samples_D_test"]
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score

error_current = 0
error_voltage = 0
smooth_order = 1

Capacity=[1]
# Resistances=[5, 10, 20, 50, 100, 10000000]
# Rates=[0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2]
# Resistances=[20, 500, 10000000]
# Rates=[2]
errors=[0, 0]

# errors_input = [0, 0]

# t_period=600
x_num_feature = 10
# capacity_periods = [120, 240, 600, 900, 1200, 1800]
capacity_periods = [60]
# t_step_periods = [100]

# X_max = [4.339977261, 0.007368864, 7.154, 4.323243104, 0.003604036, 7.154, 4292.4, 2599.034436]
# X_min = [2.790661502, -0.000782145, -7.154, 3.139090545, -0.00232158, -7.154, -4292.4, 1798.51833]
# X_mean = [3.7538, 0.0000, 0.1184, 3.7576, 0.0000, 0.1184, 71.0692, 2253.8703]
X_max = [4.339977261, 2.191329391, 7.154, 4.382654491, 2.191329391, 7.154, 4292.4,2599.034436 ]
X_min = [-0.000419, -0.23662166, 0, -0.000419, -0.23662166, -7.154, -414.932 ,-0.0015552]
X_mean = [3.595421184, 0.00109842, 0.099146893, 3.664921519, 0.001063486, 0.073874705, 58.77931943, 1916.457517]


filename="df"
# modelname="dt"
prefix='c0_'
suffix1="_scaled_test_V3"
intput_address = "D:/ml safety risk/Project/Training_data/"

num_repeat = 1
d_f1=0

for i in range(num_repeat):
    cv=[]
    cvd=[]
    f1=[]
    model=[]
    for capacity_period in capacity_periods:

        name= '_ei'+str(error_current)+'_eu'+str(error_voltage)+'_t'+ str(capacity_period)
        model_name_fig= name
        model_name= name+'.model'

        groupname="I0"
        file=intput_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order)
        print("read: ",file)
        df=pd.read_csv(file+'.csv')

        groupname="C0"
        file=intput_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order)
        print("read: ",file)
        df2=pd.read_csv(file+'.csv')
        df=pd.concat([df,df2])

        groupname="CP0"
        file=intput_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order)
        print("read: ",file)
        df3=pd.read_csv(file+'.csv')
        df=pd.concat([df,df3])

        # print(df.filter(items=['x', 'y']))
        #
        # print(max(df.filter(items=['x']).to_numpy()))
        # print(min(df.filter(items=['x']).to_numpy()))
        # print(df.filter(items=['x']).mean())

        # X_tests=[]
        # y_tests=[]
        # X_train=[]
        # y_train=[]
        # X_test=[]
        # y_test=[]

        test_size = 0.05

        # capacity_ratio = 1
        # c_rate = [1,2]

        samples_train=[]
        samples_test=[]

        df=df[df['capacity_ratio']==1]
        # df=df[df['c_rate']>0]
        # df=pd.concat([df[df['isc_resistance']<=Resistances[-2]],df[df['isc_resistance']>=Resistances[-1]]])

        # print(df)

        df=df.sample(frac=1, random_state=i)

        cut_idx = int(round(test_size * df.shape[0]))
        group_test, group_train = pd.DataFrame(df.iloc[:cut_idx]), pd.DataFrame(df.iloc[cut_idx:])

        # print(df)

        # group_test=df.iloc[0]
        # group_train=df.iloc[1]
        #
        # df=df.sample(frac= 1.0).groupby(["isc_resistance"])
        # print(df)
        #
        # for name, group in df:
        #     cut_idx = int(round(test_size * group.shape[0]))
        #     group_test_temp, group_train_temp = group.iloc[:cut_idx], group.iloc[cut_idx:]
        #     group_test=pd.concat([group_test,group_test_temp], ignore_index=True)
        #     group_train=pd.concat([group_train,group_train_temp], ignore_index=True)

        X_train = group_train.filter(items=group_train.columns[-9:-1]).to_numpy()
        y_train = group_train.filter(items=group_train.columns[-1]).to_numpy().ravel()
        X_test = group_test.filter(items=group_test.columns[-9:-1]).to_numpy()
        y_test = group_test.filter(items=group_test.columns[-1]).to_numpy().ravel()

        # print(X_train[1][1])

        X_train=function.datascaler_V3(X=X_train, X_max=X_max, X_min=X_min, X_mean=X_mean)
        X_test=function.datascaler_V3(X=X_test, X_max=X_max, X_min=X_min, X_mean=X_mean)

        # print(X_test)
        # print(y_test)

        print("Read ",len(y_train), " training samples")
        print("Read ",len(y_test), " testing samples")
        print("start training...")
        # print(X_train[1])

        # print(function.ifelserole(X_train, y_train))

        # t0 = time.time()
        # f1.append(function.ifelserole(X_train, y_train))
        # model.append("ifelserole")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)

        # t0 = time.time()
        #
        # clf0 = DecisionTreeClassifier(random_state=i,max_depth=1, max_features = 1).fit(X_train, y_train)
        # scores = cross_val_score(clf0, X_train, y_train, cv=5, scoring='f1_macro')
        # cv.append(np.mean(scores))
        # cvd.append(np.var(scores))
        # f1.append(f1_score(clf0.predict(X_test), y_test, average='macro'))
        # # joblib.dump(clf0, "dt1"+suffix1+model_name)
        # model.append("dt1")
        # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)



        # t0 = time.time()
        # clf1 = LogisticRegression(random_state=i,
        #                           class_weight='balanced',
        #                           ).fit(X_train, y_train)
        # # scores = cross_val_score(clf1, X_train, y_train, cv=5, scoring='f1_macro')
        # # cv.append(np.mean(scores))
        # # cvd.append(np.var(scores))
        # f1.append(f1_score(clf1.predict(X_test), y_test, average='macro'))
        # joblib.dump(clf1, prefix+"log"+suffix1+model_name)
        # model.append("log")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)
        #
        #
        # t0 = time.time()
        # clf2 = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree').fit(X_train, y_train)
        # # scores = cross_val_score(clf2, X_train, y_train, cv=5, scoring='f1_macro')
        # # cv.append(np.mean(scores))
        # # cvd.append(np.var(scores))
        # f1.append(f1_score(clf2.predict(X_test), y_test, average='macro'))
        # joblib.dump(clf2, prefix+"knn"+suffix1+model_name)
        # model.append("knn")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)
        #
        # t0 = time.time()
        # clf3 = DecisionTreeClassifier(random_state=i,class_weight='balanced').fit(X_train, y_train)
        # # scores = cross_val_score(clf3, X_train, y_train, cv=5, scoring='f1_macro')
        # # cv.append(np.mean(scores))
        # # cvd.append(np.var(scores))
        # f1.append(f1_score(clf3.predict(X_test), y_test, average='macro'))
        # joblib.dump(clf3, prefix+"dt"+suffix1+model_name)
        # model.append("dt")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)
        #
        # t0 = time.time()
        # clf4 = RandomForestClassifier( n_estimators=20,min_samples_split=2,criterion='entropy',class_weight='balanced',bootstrap=False).fit(X_train, y_train)
        # # scores = cross_val_score(clf4, X_train, y_train, cv=5, scoring='f1_macro')
        # # cv.append(np.mean(scores))
        # # cvd.append(np.var(scores))
        # f1.append(f1_score(clf4.predict(X_test), y_test, average='macro'))
        # joblib.dump(clf4, prefix+"rfc"+suffix1+model_name)
        # model.append("rfc")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)

        t0 = time.time()
        clf5 = svm.SVC(kernel='rbf',C=1,class_weight='balanced').fit(X_train, y_train)
        joblib.dump(clf5, "svc"+suffix1+model_name)
        # scores = cross_val_score(clf5, X_train, y_train, cv=5, scoring='f1_macro')
        # cv.append(np.mean(scores))
        # cvd.append(np.var(scores))
        f1.append(f1_score(clf5.predict(X_test), y_test, average='macro'))
        model.append("svc")
        # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        print(model[-1],f1[-1],time.time() - t0)

        #
        # t0 = time.time()
        # clf10 = RidgeClassifier().fit(X_train, y_train)
        # scores = cross_val_score(clf10, X_train, y_train, cv=5, scoring='f1_macro')
        # cv.append(np.mean(scores))
        # cvd.append(np.var(scores))
        # f1.append(f1_score(clf10.predict(X_test), y_test, average='macro'))
        # # joblib.dump(clf10, "rc"+suffix1+model_name)
        # model.append("rc")
        # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)

        # clf5 = svm.SVC(kernel='rbf',
        #                C=1e4,
        #                gamma=10000,
        #                class_weight='balanced',
        #                # tol=0.001
        #                ).fit(X_train, y_train)
        # f1.append(f1_score(clf5.predict(X_test), y_test, average='weighted'))
        # joblib.dump(clf5, "svc"+suffix1+model_name)
        # model.append("svc")
        # print(model[-1],f1[-1])


        # kernel = 1.0 * RBF(1.0)
        # clf6 = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        # f1.append(f1_score(clf6.predict(X_test), y_test, average='weighted'))
        # joblib.dump(clf6, "gpc_5_100_0_0_300.model")

        # clf7 = GaussianNB().fit(X_train, y_train)
        # f1.append(f1_score(clf7.predict(X_test), y_test, average='weighted'))
        # joblib.dump(clf7, "gnb"+suffix1+model_name)
        # model.append("gnb")
        # print(model[-1],f1[-1])
        #
        # clf_gnb = GaussianNB()
        # clf8 = CalibratedClassifierCV(clf_gnb, cv=5, method="isotonic")
        # clf8.fit(X_train, y_train)
        # f1.append(f1_score(clf8.predict(X_test), y_test, average='weighted'))
        # joblib.dump(clf8, "gnb_isotonic"+suffix1+model_name)
        # model.append("gnb_isotonic")
        # print(model[-1],f1[-1])
        #
        # clf9 = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
        # f1.append(f1_score(clf9.predict(X_test), y_test, average='weighted'))
        # joblib.dump(clf9, "qda"+suffix1+model_name)
        # model.append("qda")
        # print(model[-1],f1[-1])

        # print(f1)
        # print("Finished,", time.time() - t0, " seconds process time")

        # class_names={"Normal","Defective"}
        # disp1 = metrics.plot_confusion_matrix(clf1, X_test, y_test,
        #                                       # display_labels=class_names,
        #                                       normalize=None)
        # disp1.figure_.suptitle("Confusion Matrix")
        # print(f"Confusion matrix:\n{disp1.confusion_matrix}")
        # # plt.savefig('cm'+model_name_fig+'.jpg')
        # plt.show()