from numpy import *
import numpy as np
from sklearn import svm
import time
import joblib
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

smooth_order = 1
Resistances=[5, 10, 20, 50, 100, 10000000]

error_current = 0
error_voltage = 0

x_num_feature = 10
capacity_periods = [300]

X_max = [4.339977261, 0.007368864, 7.154, 4.323243104, 0.003604036, 7.154, 4292.4, 2599.034436]
X_min = [2.790661502, -0.000782145, -7.154, 3.139090545, -0.00232158, -7.154, -4292.4, 1798.51833]
X_mean = [3.7538, 0.0000, 0.1184, 3.7576, 0.0000, 0.1184, 71.0692, 2253.8703]

filename="df"
suffix1="_scaled_test_V3"
intput_address = "D:/ml safety risk/Project/Training_data/"

groupname="C"
test_size = 0.05
num_repeat = 5
d_f1=0

def datascaler(X=None, X_max=None, X_min=None, X_mean=None):
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            X[i][j]=(X[i][j]-X_mean[j])/(X_max[j]-X_min[j])
    return X

print("start")

for i in range(num_repeat):
    f1=[]
    cv=[]
    cvd=[]
    model=[]
    for capacity_period in capacity_periods:

        name= '_r'+str(Resistances[0])+'_'+str(Resistances[-2])+'_ei'+str(error_current)+'_eu'+str(error_voltage)+'_t'+ str(capacity_period)
        model_name_fig= name
        model_name= name+'.model'

        file=intput_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order)
        print("read: ",file)

        df=pd.read_csv(file+'.csv')

        samples_train=[]
        samples_test=[]

        df=df[df['capacity_ratio']==1]
        df=df[df['c_rate']>0]
        df=pd.concat([df[df['isc_resistance']==Resistances[0]], df[df['isc_resistance']==Resistances[-2]],df[df['isc_resistance']==Resistances[-1]]])

        df=df.sample(frac=1, random_state=i)

        cut_idx = int(round(test_size * df.shape[0]))
        group_test, group_train = pd.DataFrame(df.iloc[:cut_idx]), pd.DataFrame(df.iloc[cut_idx:])

        X_train = group_train.filter(items=group_train.columns[-9:-1]).to_numpy()
        y_train = group_train.filter(items=group_train.columns[-1]).to_numpy().ravel()
        X_test = group_test.filter(items=group_test.columns[-9:-1]).to_numpy()
        y_test = group_test.filter(items=group_test.columns[-1]).to_numpy().ravel()

        X_train=datascaler(X=X_train, X_max=X_max, X_min=X_min, X_mean=X_mean)
        X_test=datascaler(X=X_test, X_max=X_max, X_min=X_min, X_mean=X_mean)

        print("Read ",len(y_train), " training samples")
        print("Read ",len(y_test), " testing samples")
        print("start training...")
        
        t0 = time.time()
        clf5 = svm.SVC(kernel='rbf',C=1e4,gamma=1e4,class_weight='balanced').fit(X_train, y_train)
        # joblib.dump(clf5, prefix+"svc"+suffix1+model_name)
        # scores = cross_val_score(clf5, X_train, y_train, cv=5, scoring='f1_macro')
        # cv.append(np.mean(scores))
        # cvd.append(np.var(scores))
        f1.append(f1_score(clf5.predict(X_test), y_test, average='macro'))
        model.append("svc")
        # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        print(model[-1],f1[-1],time.time() - t0)


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

        # t0 = time.time()
        # clf10 = RidgeClassifier().fit(X_train, y_train)
        # scores = cross_val_score(clf10, X_train, y_train, cv=5, scoring='f1_macro')
        # cv.append(np.mean(scores))
        # cvd.append(np.var(scores))
        # f1.append(f1_score(clf10.predict(X_test), y_test, average='macro'))
        # # joblib.dump(clf10, "rc"+suffix1+model_name)
        # model.append("rc")
        # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)


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
