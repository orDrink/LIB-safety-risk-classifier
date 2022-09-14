import function
from numpy import *
from sklearn import svm
import time
import joblib
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

error_current = 0
error_voltage = 0
smooth_order = 1
Capacity=[1]
errors=[0, 0]
x_num_feature = 10
capacity_periods = [60]

X_max = [4.339977261, 2.191329391, 7.154, 4.382654491, 2.191329391, 7.154, 4292.4,2599.034436 ]
X_min = [-0.000419, -0.23662166, 0, -0.000419, -0.23662166, -7.154, -414.932 ,-0.0015552]
X_mean = [3.595421184, 0.00109842, 0.099146893, 3.664921519, 0.001063486, 0.073874705, 58.77931943, 1916.457517]

filename="df"
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

        test_size = 0.05
        samples_train=[]
        samples_test=[]
        df=df[df['capacity_ratio']==1]
        df=df.sample(frac=1, random_state=i)
        cut_idx = int(round(test_size * df.shape[0]))
        group_test, group_train = pd.DataFrame(df.iloc[:cut_idx]), pd.DataFrame(df.iloc[cut_idx:])

        X_train = group_train.filter(items=group_train.columns[-9:-1]).to_numpy()
        y_train = group_train.filter(items=group_train.columns[-1]).to_numpy().ravel()
        X_test = group_test.filter(items=group_test.columns[-9:-1]).to_numpy()
        y_test = group_test.filter(items=group_test.columns[-1]).to_numpy().ravel()

        X_train=function.datascaler_V3(X=X_train, X_max=X_max, X_min=X_min, X_mean=X_mean)
        X_test=function.datascaler_V3(X=X_test, X_max=X_max, X_min=X_min, X_mean=X_mean)

        print("Read ",len(y_train), " training samples")
        print("Read ",len(y_test), " testing samples")
        print("start training...")

        # t0 = time.time()
        # f1.append(function.ifelserole(X_train, y_train))
        # model.append("ifelserole")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)

        t0 = time.time()
        
        clf0 = DecisionTreeClassifier(random_state=i,max_depth=1, max_features = 1).fit(X_train, y_train)
        scores = cross_val_score(clf0, X_train, y_train, cv=5, scoring='f1_macro')
        cv.append(np.mean(scores))
        cvd.append(np.var(scores))
        f1.append(f1_score(clf0.predict(X_test), y_test, average='macro'))
        # joblib.dump(clf0, "dt1"+suffix1+model_name)
        model.append("dt1")
        print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)


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

        # t0 = time.time()
        # clf5 = svm.SVC(kernel='rbf',C=1,class_weight='balanced').fit(X_train, y_train)
        # joblib.dump(clf5, "svc"+suffix1+model_name)
        # # scores = cross_val_score(clf5, X_train, y_train, cv=5, scoring='f1_macro')
        # # cv.append(np.mean(scores))
        # # cvd.append(np.var(scores))
        # f1.append(f1_score(clf5.predict(X_test), y_test, average='macro'))
        # model.append("svc")
        # # print(model[-1],cv[-1],cvd[-1],f1[-1],time.time() - t0)
        # print(model[-1],f1[-1],time.time() - t0)

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
