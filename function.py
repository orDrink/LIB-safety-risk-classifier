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
import random

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

def read_cnn(filelist, parameters, Capacity=None, Resistances=None, Rates=None):
    data_X = []
    data_Y = []
    for filename in filelist:
        # results = open(filename + ".csv", "r")
        # reader_results = csv.reader(results)

        reader_results =np.loadtxt(open(filename + ".csv","rb"),delimiter=",",skiprows=0)
        # print(filename)
        # random.shuffle(reader_results)

        for item in reader_results:
            xl=len(item)-1
            data_X_temp = []
            if (float(item[0]) in Capacity) and (float(item[1]) in Resistances) and (float(item[2]) in Rates):
                for i in range(xl-parameters):
                    data_X_temp.append(float(item[i+parameters]))
                data_X.append(data_X_temp)
                data_Y.append(float(item[xl]))
        # results.close()
    X = np.array(data_X)
    Y = np.array(data_Y)

    return datascaler_cnn(X,Y)
    # return X,Y

def datascaler_cnn(X,Y):
    max_abs_scaler = preprocessing.MaxAbsScaler()

    I_max = 7.3
    I_min = -7.3
    U_max = 4.4
    U_min = 2.5
    dU_max = 0.01
    dU_min = 0
    features = 10
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if j // features == 0:
                X[i][j]=(X[i][j]-U_min)/(U_max-U_min)
            if j // features == 1:
                X[i][j]=(X[i][j]-dU_min)/(dU_max-dU_min)
            if j // features == 2:
                X[i][j]=(X[i][j]-I_min)/(I_max-I_min)

    return X,Y

def read_rf(filelist, parameters, Capacity=None, Resistances=None, Rates=None):
    data_X = []
    data_Y = []
    for filename in filelist:
        # results = open(filename + ".csv", "r")
        # reader_results = csv.reader(results)

        reader_results =np.loadtxt(open(filename + ".csv","rb"),delimiter=",",skiprows=0)
        # print(filename)
        # random.shuffle(reader_results)

        for item in reader_results:
            xl=len(item)-1
            data_X_temp = []
            if (float(item[0]) in Capacity):
                if (float(item[1]) in Resistances) :
                    if (float(item[2]) in Rates):
                        for i in range(xl-parameters):
                            data_X_temp.append(float(item[i+parameters]))
                        data_X.append(data_X_temp)
                        data_Y.append(float(item[xl]))
        # results.close()
    X = np.array(data_X)
    Y = np.array(data_Y)

    return X,Y

def datascaler_rf(X=None, X_max=None, X_min=None):
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            X[i][j]=(X[i][j]-X_max[j])/(X_max[j]-X_min[j])
    return X

def datascaler_V3(X=None, X_max=None, X_min=None, X_mean=None):
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            X[i][j]=(X[i][j]-X_mean[j])/(X_max[j]-X_min[j])
    return X

def datascaler_rf0(X,Y):
    max_abs_scaler = preprocessing.MaxAbsScaler()
    I_max = 7.3
    I_min = -7.3
    U_max = 4.4
    U_min = 2.5
    Q_min = 0
    Q_max = 6000
    dU_max = 0.01
    dU_min = 0
    features = 10
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if j // features == 0:
                X[i][j]=(X[i][j]-U_min)/(U_max-U_min)
            if j // features == 1:
                X[i][j]=(X[i][j]-dU_min)/(dU_max-dU_min)
            if j // features == 2:
                X[i][j]=(X[i][j]-Q_min)/(Q_max-Q_min)
            if j // features == 3:
                X[i][j]=(X[i][j]-I_min)/(I_max-I_min)
    return X,Y

def read_rf_model1(filelist, num_feature = 8):
    data_X = []
    data_Y = []

    for filename in filelist:
        reader_results =np.loadtxt(open(filename + ".csv","rb"),delimiter=",",skiprows=0)
        for item in reader_results:
            xl=len(item)-1
            data_X_temp = []
            parameters = xl - num_feature
            for i in range(num_feature):
                 data_X_temp.append(float(item[i+parameters]))
            data_X.append(data_X_temp)
            data_Y.append(float(item[xl]))
        # results.close()
    X = np.array(data_X)
    Y = np.array(data_Y)
    return X,Y

def read_group0(filelist, parameters, capacity):
    data_X = []
    data_Y = []
    for filename in filelist:
        results = open(filename + ".csv", "r")
        reader_results = csv.reader(results)
        # print(filename)

        for item in reader_results:
            xl=len(item)-1
            data_X_temp = []
            if (int(item[xl]) == 0):
                if (float(item[0]) == capacity) and (float(item[1]) > 10000):
                    for i in range(xl-parameters):
                        data_X_temp.append(float(item[i+parameters]))
                    data_X.append(data_X_temp)
                    data_Y.append(float(item[xl]))

        results.close()

    X = np.array(data_X)
    Y = np.array(data_Y)

    return datascaler(X,Y)
    # return X,Y

def read_group1(filelist, parameters, resistance, capacity):
    data_X = []
    data_Y = []
    for filename in filelist:
        results = open(filename + ".csv", "r")
        reader_results = csv.reader(results)
        # print(filename)

        for item in reader_results:
            xl=len(item)-1
            data_X_temp = []

            if (int(item[xl]) == 1):
                if (float(item[0]) == capacity) and (int(item[1]) == resistance):
                    for i in range(xl-parameters):
                        data_X_temp.append(float(item[i+parameters]))
                    data_X.append(data_X_temp)
                    data_Y.append(float(item[xl]))
        results.close()

    X = np.array(data_X)
    Y = np.array(data_Y)

    return datascaler(X,Y)
    # return X,Y

def read_group2(filelist, dimension_X):
    data_X = []
    data_Y = []
    for filename in filelist:
        results = open(filename + ".csv", "r")
        reader_results = csv.reader(results)
        # print(filename)

        for item in reader_results:
            parameters = len(item)-dimension_X
            xl=len(item)-1
            data_X_temp = []
            if (float(item[xl])==2):
                for i in range(xl-parameters):
                    data_X_temp.append(float(item[i+parameters]))
                data_X.append(data_X_temp)
                data_Y.append(float(item[xl]))
        results.close()

    X = np.array(data_X)
    Y = np.array(data_Y)

    return datascaler(X,Y)
    # return X,Y

def read_group3(filelist, dimension_X):
    data_X = []
    data_Y = []
    for filename in filelist:
        results = open(filename + ".csv", "r")
        reader_results = csv.reader(results)
        # print(filename)

        for item in reader_results:
            parameters = len(item)-dimension_X
            xl=len(item)-1
            data_X_temp = []
            if (float(item[xl])==3):
                for i in range(xl-parameters):
                    data_X_temp.append(float(item[i+parameters]))
                data_X.append(data_X_temp)
                data_Y.append(float(item[xl]))
        results.close()

    X = np.array(data_X)
    Y = np.array(data_Y)

    return datascaler(X,Y)
    # return X,Y

def smmoth(x,y, window_size=10, polynomial_order=3):
    yhat = savgol_filter(y, window_size, polynomial_order)

def error_X(X_sampeles, time_step=10, current_error=0.001, voltage_error=0.001):
    for X in X_sampeles:
        for i in range(len(X)-1):
            if i==0:
                X[i]=X[i] * (1+random.uniform(-current_error/2, current_error/2))
            elif (i==1):
                X[i]=X[i] * (1+random.uniform(-voltage_error/2, voltage_error/2))
            elif (i>1) and (i<len(X)-1):
                X[i]=X[i] * (1+random.uniform(-voltage_error/2, voltage_error/2))
            else:
                X[i]=X[i]
    return X_sampeles

def datascaler(X,Y):
    max_abs_scaler = preprocessing.MaxAbsScaler()

    I_ref = 7.3
    I_min = 0
    U_max = 4.4
    U_min = 2.5
    dU_ref1 = 0.0097
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            if j % 3 == 2:
                X[i][j]=(X[i][j]-I_min)/(I_ref-I_min)
            elif j % 3 == 0:
                X[i][j]=(X[i][j]-U_min)/(U_max-U_min)
            elif j % 3 == 1:
                X[i][j]=X[i][j]/dU_ref1
            # elif j == 3:
            #     X[i][j]=X[i][j]/dU_ref2

    # return max_abs_scaler.fit_transform(X),Y
    return X,Y


def process_bar(time_used, time_left, percent, start_str='|', end_str='100%', total_length=50):
    bar = ''.join(['▋'] * int(percent * total_length)) + ''
    # "\033[31m%s\033[0m"
    bar = '\r' + start_str + bar.ljust(total_length) + '|{:0>4.1f}%|'.format(percent*100) + end_str
    print(bar," |used",'%.2f' % time_used,"mins |left", '%.2f' % time_left,"mins", end='', flush=True)

def process_bar_1(percent, start_str='|', end_str='100%', total_length=50):
    bar = ''.join(['▋'] * int(percent * total_length)) + ''
    bar = '\r'+ start_str + bar.ljust(total_length) + '|'+ ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)

def in_value(t1, t2, y1, y2, t_in):
    y_insert = y1 + (y2 - y1) / (t2 - t1) * (t1 - t_in)
    return (y_insert)

def int_array(t, ts, vs, cs):
    t_min=min(ts)
    t_max=max(ts)
    n_step=len(ts)
    U = 0
    I = 0
    j = 0
    for j in range(n_step-1):
        if (ts[j]<=t) & (ts[j+1]>t):
            U = vs[j] + (vs[j + 1] - vs[j]) / (ts[j + 1] - ts[j]) * (t - ts[j])
            I = cs[j] + (cs[j + 1] - cs[j]) / (ts[j + 1] - ts[j]) * (t - ts[j])
    return (U,I)

def int_array_list(t_step, ts, vs, cs):
    data_int = []
    t_max = max(ts)
    t_min = min(ts)
    t = t_min
    while (t < t_max):
        # ut, it = int_array(t, ts, vs, cs)
        n_step = len(ts)
        U = 0
        I = 0
        jn = 0
        jm = jn
        for j in range(jm, n_step - 1):
            if (ts[j] <= t) & (ts[j + 1] > t):
                jn=j
                U = vs[j] + (vs[j + 1] - vs[j]) / (ts[j + 1] - ts[j]) * (t - ts[j])
                I = cs[j] + (cs[j + 1] - cs[j]) / (ts[j + 1] - ts[j]) * (t - ts[j])
        data_int.append([t, U, I])
        t = t + t_step
    return (np.array(data_int))

def interpolation_list(step=1, X=None, Y=None):
    X_int = []
    Y_int = []
    X_max = max(X[:])
    X_min = min(X[:])
    t = X_min
    print(t)
    while (t < X_max):
        # ut, it = int_array(t, ts, vs, cs)
        n_step = len(X)
        jn = 0
        jm = jn
        Yi = 0
        for j in range(jm, n_step - 1):
            if (X[j] <= t) & (X[j + 1] > t):
                jn = j
                Yi = Y[j] + (Y[j + 1] - Y[j]) / (X[j + 1] - X[j]) * (t - X[j])
        X_int.append(t)
        Y_int.append(Yi)
        t = t + step
    return (np.array(X_int),np.array(Y_int))

def divide(input_address="", output_address="", groupname="C", parameters=3, datasets=None):

    data_list=[]
    for dataset in datasets:
        results = open(input_address + dataset + ".csv", "r")
        reader_results = csv.reader(results)
        data_list=data_list+list(reader_results)

    data = np.array(data_list)
    l = data.shape[0]
    # p = data.shape[1]

    listfile = open(output_address + groupname+"_filelist.csv", "w", encoding='utf-8', newline='')
    writer_list = csv.writer(listfile)

    filenumber = 1
    parameterlist=[]
    samplefile = open(output_address + groupname+str(filenumber) + ".csv", "w", encoding='utf-8', newline='')
    parameterlist.append(filenumber)

    for k in range(parameters):
        parameterlist.append(data[0][k])
    writer_list.writerow(parameterlist)

    t_step_measure = int(float(data[1][parameters]) - float(data[0][parameters]))

    writer = csv.writer(samplefile)
    create = 0
    t0 = time.time()

    for i in range(l):
        if i > 1:
            for j in range(parameters):
                if data[i][j]!=data[i-1][j]:
                    samplefile.close()
                    filenumber = filenumber + 1

                    parameterlist=[]
                    parameterlist.append(filenumber)
                    for k in range(parameters):
                        parameterlist.append(data[i][k])
                    writer_list.writerow(parameterlist)

                    samplefile = open(output_address + groupname+str(filenumber) + ".csv", "w", encoding='utf-8', newline='')
                    writer = csv.writer(samplefile)
                    break
        if (float(data[i][parameters]) % t_step_measure == 0):
            writer.writerow(data[i])

        finished = (i+1)/l
        # process_bar_1(finished, '|', '100%', 50)
        time_used = (time.time() - t0)/60
        time_left =time_used / finished * (1-finished)
        process_bar(time_used,time_left, finished, '|', '100%', 50)

    samplefile.close()
    # writer_list.close()
    return 0

def generate_RF(input_address, output_address, groupname, parameters, t_points, t_step_measure, t_start, t_step_samples, time=None, error_current = 0, error_voltage = 0, smooth_order = 3, window_size=20):

    t0 = time.time()
    samplefile_list = open(input_address+groupname+"_filelist.csv", "r")
    readerlist = csv.reader(samplefile_list)
    numberlist = np.array(list(readerlist))

    t_period=t_points[-1]
    t_step_period = t_points[1]-t_points[0]

    samples_N = 0
    samplefile_N = open(output_address+"rf_samples_"+ groupname+ "_" + str(int(t_period)) + "_" + str(int(t_step_period))+"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order) +".csv", "w", encoding='utf-8', newline='')

    number_start=431

    for number in range(numberlist.shape[0]):
        if (number> number_start):
            # if (number> 500) and (number< 502):
            # results = open("D:/ml safety risk/Project/Splited_data/"+groupname +str(number+1) + ".csv", "r")
            # reader_results = csv.reader(results)

            array_data_int=np.loadtxt(open(input_address+groupname +str(number+1) + ".csv","rb"),delimiter=",",skiprows=0)

            t_step_measure = int(array_data_int[1][parameters]-array_data_int[0][parameters])

            T_c = 200
            T_max = 0
            Y = 0
            if groupname == "C":
                if int(numberlist[number][2])<10000:
                    Y = 1

            if groupname == "I":
                Y = 2
                T_max = max(array_data_int[:, 3])
                if T_max> T_c:
                    Y = 3

            # nt = round(min(array_data_int[:, 0]))+round((t_period+t_start)/t_step_measure)+smooth_order*2
            nt = round(min(array_data_int[:, 0]))

            writer = csv.writer(samplefile_N)

            sample_N = 0

            U_temps=[]
            I_temps=[]
            U1_temps=[]
            I1_temps=[]
            dU_temps=[]
            dI_temps=[]
            Q_temps=[0]
            Q_temp=0
            m_points= np.arange(0,int(window_size)*t_step_measure,t_step_measure)

            while (nt < len(array_data_int[:, 0])):

                U_temps.append(array_data_int[nt, parameters+1]+ np.random.normal(0, error_voltage/3))
                I_temps.append(array_data_int[nt, parameters+2]+ np.random.normal(0, error_current/3))
                Q_temps.append(float(U_temps[-1])*t_step_measure+Q_temps[-1])

                if len(I_temps)> window_size:
                    U_temps_m=[]
                    I_temps_m=[]
                    nm= len(I_temps)-1

                    for m in range(window_size):
                        U_temps_m.append(float(U_temps[nm-m]))
                        I_temps_m.append(float(I_temps[nm-m]))

                    coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
                    coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

                    dU_temps.append(coeff1[-2])
                    U1_temps.append(coeff1[-1])
                    dI_temps.append(coeff2[-2])
                    I1_temps.append(coeff2[-1])
                if len(I1_temps)> round((t_period)/t_step_measure)+1:
                    sample = []
                    dQ_temp = 0
                    dQ=[]
                    for p in range(parameters):
                        sample.append(numberlist[number][p+1])
                    ns= len(I1_temps)-1
                    for k in t_points:
                        sample.append(U1_temps[ns-int(k/ t_step_measure)])
                    for k in t_points:
                        sample.append(dU_temps[ns-int(k/ t_step_measure)])
                    for k in t_points:
                        sample.append(Q_temps[ns]-Q_temps[ns-int(k/ t_step_measure)])
                    sample.append(I1_temps[ns])
                    sample.append(Y)

                    C_rate = float(numberlist[number][3])
                    if ((array_data_int[nt, parameters+1]<4.34) and (array_data_int[nt - int(t_points[-1] / t_step_measure)- window_size,
                        parameters+1]< 4.34) and (abs(array_data_int[nt,
                        parameters+2])> C_rate*3.577*0.9) and (abs(array_data_int[nt - int(t_points[-1] / t_step_measure) - window_size,
                        parameters+2]) > C_rate*3.577*0.9) and  ((array_data_int[nt - int(t_points[-1] / t_step_measure) - window_size,
                        parameters+2]) * (array_data_int[nt ,parameters+2]) > 0)):

                        writer.writerow(sample)
                        sample_N = sample_N + 1
                        samples_N = samples_N + 1

                # if Y < 2:
                #     nt = nt + max(int(t_step_samples[Y] / t_step_measure / float(numberlist[number][3])), 1)  #scale sample generation time step
                # else:
                #     nt = nt + max(int(t_step_samples[Y] / t_step_measure), 1)

                nt = nt + max(int(t_step_samples[Y] / t_step_measure), 1)

            # print("add group ", groupname + str(number+1), ", sample number:", sample_N, ", parameters: ", numberlist[number],", Y = ", Y)

            finished = (number-number_start+1)/(numberlist.shape[0]-number_start)
            time_used = (time.time() - t0)/60
            time_left =time_used / finished * (1-finished)

            # print("finished ", '%.2f' % finished,"%", "used",'%.2f' % time_used," minutes","about",'%.2f' % time_left," minutes left")

            process_bar(time_used,time_left, finished, '|', '100%', 50)
            print(" |P=",numberlist[number], "|Y=",Y,"|N=",sample_N, end='', flush=True)
            # results.close()

    samplefile_N.close()
    print(" |finished,", "sample number:", samples_N)
    return 0

def generate_CNN(input_address, output_address, groupname, parameters, t_points, t_step_measure, t_start, t_step_samples, time=None, error_current = 0, error_voltage = 0, smooth_order = 3, window_size=20):

    t0 = time.time()
    samplefile_list = open(input_address+groupname+"_filelist.csv", "r")
    readerlist = csv.reader(samplefile_list)
    numberlist = np.array(list(readerlist))

    t_period=t_points[-1]
    t_step_period = t_points[1]-t_points[0]

    samples_N = 0
    samplefile_N = open(output_address+"cnn_samples_"+ groupname+ "_" + str(int(t_period)) + "_" + str(int(t_step_period))+"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order) +".csv", "w", encoding='utf-8', newline='')

    number_start=431

    for number in range(numberlist.shape[0]):
        if (number> number_start):
            # if (number> 500) and (number< 502):
            # results = open("D:/ml safety risk/Project/Splited_data/"+groupname +str(number+1) + ".csv", "r")
            # reader_results = csv.reader(results)

            array_data_int=np.loadtxt(open(input_address+groupname +str(number+1) + ".csv","rb"),delimiter=",",skiprows=0)

            t_step_measure = int(array_data_int[1][parameters]-array_data_int[0][parameters])

            T_c = 200
            T_max = 0
            Y = 0
            if groupname == "C":
                if int(numberlist[number][2])<10000:
                    Y = 1

            if groupname == "I":
                Y = 2
                T_max = max(array_data_int[:, 3])
                if T_max> T_c:
                    Y = 3

            # nt = round(min(array_data_int[:, 0]))+round((t_period+t_start)/t_step_measure)+smooth_order*2
            nt = round(min(array_data_int[:, 0]))

            writer = csv.writer(samplefile_N)

            sample_N = 0

            U_temps=[]
            I_temps=[]
            U1_temps=[]
            I1_temps=[]
            dU_temps=[]
            dI_temps=[]
            m_points= np.arange(0,int(window_size)*t_step_measure,t_step_measure)

            while (nt < len(array_data_int[:, 0])):

                U_temps.append(array_data_int[nt, parameters+1]+ np.random.normal(0, error_voltage/3))
                I_temps.append(array_data_int[nt, parameters+2]+ np.random.normal(0, error_current/3))

                if len(I_temps)> window_size:
                    U_temps_m=[]
                    I_temps_m=[]
                    nm= len(I_temps)-1

                    for m in range(window_size):
                        U_temps_m.append(float(U_temps[nm-m]))
                        I_temps_m.append(float(I_temps[nm-m]))

                    coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
                    coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

                    dU_temps.append(coeff1[-2])
                    U1_temps.append(coeff1[-1])
                    dI_temps.append(coeff2[-2])
                    I1_temps.append(coeff2[-1])
                if len(I1_temps)> round((t_period)/t_step_measure)+1:
                    sample = []
                    for p in range(parameters):
                        sample.append(numberlist[number][p+1])
                    ns= len(I1_temps)-1
                    for k in t_points:
                        sample.append(U1_temps[ns-int(k/ t_step_measure)])
                    for k in t_points:
                        sample.append(dU_temps[ns-int(k/ t_step_measure)])
                    for k in t_points:
                        sample.append(I1_temps[ns-int(k/ t_step_measure)])
                    sample.append(Y)

                    C_rate = float(numberlist[number][3])
                    if (array_data_int[nt, parameters+1]<4.34) and (array_data_int[nt - int(t_points[-1] / t_step_measure)- window_size,
                        parameters+1]< 4.34) and (abs(array_data_int[nt,
                        parameters+2])> C_rate*3.577*0.9) and (abs(array_data_int[nt - int(t_points[-1] / t_step_measure) - window_size,
                        parameters+2]) > C_rate*3.577*0.9) and  ((array_data_int[nt - int(t_points[-1] / t_step_measure) - window_size,
                        parameters+2]) * (array_data_int[nt ,parameters+2]) > 0):

                        writer.writerow(sample)
                        sample_N = sample_N + 1
                        samples_N = samples_N + 1

                # if Y < 2:
                #     nt = nt + max(int(t_step_samples[Y] / t_step_measure / float(numberlist[number][3])), 1)  #scale sample generation time step
                # else:
                #     nt = nt + max(int(t_step_samples[Y] / t_step_measure), 1)

                nt = nt + max(int(t_step_samples[Y] / t_step_measure), 1)

            # print("add group ", groupname + str(number+1), ", sample number:", sample_N, ", parameters: ", numberlist[number],", Y = ", Y)

            finished = (number-number_start+1)/(numberlist.shape[0]-number_start)
            time_used = (time.time() - t0)/60
            time_left =time_used / finished * (1-finished)

            # print("finished ", '%.2f' % finished,"%", "used",'%.2f' % time_used," minutes","about",'%.2f' % time_left," minutes left")

            process_bar(time_used,time_left, finished, '|', '100%', 50)
            print(" |P=",numberlist[number], "|Y=",Y,"|N=",sample_N, end='', flush=True)
            # results.close()

    samplefile_N.close()
    print(" |finished,", "sample number:", samples_N)
    return 0

def generate_Normalized(input_address="", output_address="", groupname="C", filename="", parameters=3, capacity_period=600 , t_points=None, t_step_measure=10, t_step_samples=None, time=None, error_current = 0, error_voltage = 0, smooth_order = 3, window_size=20, number_start = 431):

    # if t_points is None:
    #     t_points = [0]
    t0 = time.time()
    samplefile_list = open(input_address+groupname+"_filelist.csv", "r")
    readerlist = csv.reader(samplefile_list)
    numberlist = np.array(list(readerlist))

    # t_step_period = t_points[1]-t_points[0]

    samples_N = 0
    samplefile_N = open(output_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order) +".csv", "w", encoding='utf-8', newline='')

    # number_start=431

    for number in range(numberlist.shape[0]):
        if (number> number_start):
            # if (number> 500) and (number< 502):
            # results = open("D:/ml safety risk/Project/Splited_data/"+groupname +str(number+1) + ".csv", "r")
            # reader_results = csv.reader(results)

            array_data_int=np.loadtxt(open(input_address+groupname +str(number+1) + ".csv","rb"),delimiter=",",skiprows=0)

            t_step_measure = int(array_data_int[1][parameters]-array_data_int[0][parameters])

            T_c = 200
            T_max = 0
            Y = 0
            window_size_r = int(window_size)
            t_period=int(capacity_period)

            if groupname == "C" or groupname == "EC" :
                t_period=int(capacity_period/float(numberlist[number][3]))
                window_size_r = int(window_size/float(float(numberlist[number][3])))
                if int(numberlist[number][2])<10000:
                    Y = 1

            if groupname == "I":
                Y = 2
                T_max = max(array_data_int[:, 3])
                if T_max> T_c:
                    Y = 3

            # nt = round(min(array_data_int[:, 0]))+round((t_period+t_start)/t_step_measure)+smooth_order*2
            nt = round(min(array_data_int[:, 0]))

            writer = csv.writer(samplefile_N)

            sample_N = 0

            U_temps=[]
            I_temps=[]
            U1_temps=[]
            I1_temps=[]
            dU_temps=[]
            dI_temps=[]
            Q_temps=[0]
            Q_leak_temps=[0]
            Q_temp=0


            m_points= np.arange(0,int(window_size_r)*t_step_measure,t_step_measure)

            while (nt < len(array_data_int[:, 0])):

                U_temps.append(array_data_int[nt, parameters+1]+ np.random.normal(0, error_voltage/3))
                I_temps.append(array_data_int[nt, parameters+2]+ np.random.normal(0, error_current/3))
                Q_temps.append(float(I_temps[-1])*t_step_measure+Q_temps[-1])
                Q_leak_temps.append(float(U_temps[-1])*t_step_measure+Q_leak_temps[-1])

                if len(I_temps)> window_size_r:
                    U_temps_m=[]
                    I_temps_m=[]
                    nm= len(I_temps)-1

                    for m in range(window_size_r):
                        U_temps_m.append(float(U_temps[nm-m]))
                        I_temps_m.append(float(I_temps[nm-m]))

                    coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
                    coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

                    dU_temps.append(coeff1[-2])
                    U1_temps.append(coeff1[-1])
                    dI_temps.append(coeff2[-2])
                    I1_temps.append(coeff2[-1])
                if len(I1_temps)> round((t_period)/t_step_measure)+1:
                    sample = []
                    for p in range(parameters):
                        sample.append(numberlist[number][p+1])
                    ns= len(I1_temps)-1
                    dT= t_period
                    sample.append(U1_temps[ns])
                    sample.append(dU_temps[ns])
                    sample.append(U1_temps[ns-int(dT/ t_step_measure)])
                    sample.append(dU_temps[ns-int(dT/ t_step_measure)])
                    sample.append(I1_temps[ns])
                    sample.append(Q_temps[ns]-Q_temps[ns-int(dT/ t_step_measure)])
                    sample.append(Q_leak_temps[ns]-Q_leak_temps[ns-int(dT/ t_step_measure)])
                    sample.append(Y)

                    C_rate = float(numberlist[number][3])
                    if ((array_data_int[nt, parameters+1]<4.34) and (array_data_int[nt - int(dT / t_step_measure)- window_size_r,
                        parameters+1]< 4.34) and (abs(array_data_int[nt,
                        parameters+2])> C_rate*3.577*0.9) and (abs(array_data_int[nt - int(dT / t_step_measure) - window_size_r,
                        parameters+2]) > C_rate*3.577*0.9) and  ((array_data_int[nt - int(dT / t_step_measure) - window_size_r,
                        parameters+2]) * (array_data_int[nt ,parameters+2]) > 0)):

                        writer.writerow(sample)
                        sample_N = sample_N + 1
                        samples_N = samples_N + 1

                nt = nt + 1

            # print("add group ", groupname + str(number+1), ", sample number:", sample_N, ", parameters: ", numberlist[number],", Y = ", Y)

            finished = (number-number_start+1)/(numberlist.shape[0]-number_start)
            time_used = (time.time() - t0)/60
            time_left =time_used / finished * (1-finished)

            # print("finished ", '%.2f' % finished,"%", "used",'%.2f' % time_used," minutes","about",'%.2f' % time_left," minutes left")

            process_bar(time_used,time_left, finished, '|', '100%', 50)
            print(" |P=",numberlist[number], "|Y=",Y,"|N=",sample_N, end='', flush=True)

            # results.close()

    samplefile_N.close()
    print(" |finished,", "sample number:", samples_N)
    return 0

def generate_Normalized_2(model="model2",
                          input_address="",
                          output_address="",
                          groupname="C",
                          filename="",
                          parameters=3,
                          capacity_period=600 ,
                          t_points=None,
                          t_step_measure=10,
                          t_step_samples=None,
                          isc_threshold = 0.9,
                          isc_time = 200,
                          time=None,
                          error_current = 0,
                          error_voltage = 0,
                          smooth_order = 3,
                          window_size=20,
                          number_start = 432,
                          period_normalized = True):

    # if t_points is None:
    #     t_points = [0]
    t0 = time.time()
    samplefile_list = open(input_address+groupname+"_filelist.csv", "r")
    readerlist = csv.reader(samplefile_list)
    numberlist = np.array(list(readerlist))

    # t_step_period = t_points[1]-t_points[0]

    samples_N = 0
    samplefile_N = open(output_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order) +".csv", "w", encoding='utf-8', newline='')

    # number_start=431

    for number in range(numberlist.shape[0]):

        if (number> number_start-1):

            array_data_int=np.loadtxt(open(input_address+groupname +str(number+1) + ".csv","rb"),delimiter=",",skiprows=0)

            if len(array_data_int)>capacity_period:

                t_step_measure = int(array_data_int[1][parameters]-array_data_int[0][parameters])

                T_c = 200
                T_max = 0
                Y = 0
                window_size_r = int(window_size)
                t_period=int(capacity_period)

                if groupname == "C" or groupname == "EC" :
                    if (model == 'model2'):
                        t_period=int(capacity_period/float(numberlist[number][3]))
                        window_size_r = int(window_size/(float(numberlist[number][3])))
                    if int(numberlist[number][2])<10000:
                        Y = 1

                if groupname == "I" or groupname == "EI" or groupname == "EI1":
                    Y = 2
                    T_max = max(array_data_int[:, parameters+3])
                    if T_max> T_c:
                        Y = 3

                # nt = round(min(array_data_int[:, 0]))+round((t_period+t_start)/t_step_measure)+smooth_order*2
                nt = 0

                writer = csv.writer(samplefile_N)

                sample_N = 0

                U_temps=[]
                I_temps=[]
                U1_temps=[]
                I1_temps=[]
                dU_temps=[]
                dI_temps=[]
                Q_temps=[0]
                Q_leak_temps=[0]
                Q_temp=0

                m_points= np.arange(0,int(window_size_r)*t_step_measure,t_step_measure)
                isc_start = 0

                while (nt < len(array_data_int[:, 0])):

                    U_temps.append(array_data_int[nt, parameters+1] + np.random.normal(0, error_voltage/3))
                    I_temps.append(array_data_int[nt, parameters+2] + np.random.normal(0, error_current/3))
                    Q_temps.append(float(I_temps[-1])*t_step_measure+Q_temps[-1])
                    Q_leak_temps.append(float(U_temps[-1])*t_step_measure+Q_leak_temps[-1])

                    if len(I_temps)> window_size_r and smooth_order > 0:
                        U_temps_m=[]
                        I_temps_m=[]
                        nm= len(I_temps)-1

                        for m in range(window_size_r):
                            U_temps_m.append(float(U_temps[nm-m]))
                            I_temps_m.append(float(I_temps[nm-m]))

                        coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
                        coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

                        dU_temps.append(coeff1[-2])
                        U1_temps.append(coeff1[-1])
                        dI_temps.append(coeff2[-2])
                        I1_temps.append(coeff2[-1])

                    write_point=nt % int(t_step_samples[Y]/t_step_measure)

                    if len(I1_temps)> round((t_period)/t_step_measure)+1 and ((model == 'model2') or ((model == 'model1') and (Y<2))) and write_point==0:
                        sample = []
                        for p in range(parameters):
                            sample.append(numberlist[number][p+1])
                        ns= len(I1_temps)-1
                        dT= t_period
                        sample.append(U1_temps[ns])
                        sample.append(dU_temps[ns])
                        sample.append(I1_temps[ns])
                        sample.append(U1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(dU_temps[ns-int(dT/ t_step_measure)])
                        sample.append(I1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_temps[ns]-Q_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_leak_temps[ns]-Q_leak_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Y)

                        C_rate = float(numberlist[number][3])
                        if ((array_data_int[nt, parameters+1]<4.34)
                                and (array_data_int[nt - int(dT / t_step_measure)- window_size_r, parameters+1]< 4.34)
                                and (abs(array_data_int[nt, parameters+2])> C_rate*3.577*0.9)
                                and (abs(array_data_int[nt - int(dT / t_step_measure) - window_size_r, parameters+2]) > C_rate*3.577*0.9)
                                and ((array_data_int[nt - int(dT / t_step_measure) - window_size_r, parameters+2]) * (array_data_int[nt ,parameters+2]) > 0)):
                            writer.writerow(sample)
                            sample_N = sample_N + 1
                            samples_N = samples_N + 1

                    if len(I1_temps)> round((t_period)/t_step_measure)+1  and (model == 'model1') and (Y>=2) and write_point==0:
                        sample = []
                        for p in range(parameters):
                            sample.append(numberlist[number][p+1])
                        ns= len(I1_temps)-1
                        dT= t_period
                        sample.append(U1_temps[ns])
                        sample.append(dU_temps[ns])
                        sample.append(I1_temps[ns])
                        sample.append(U1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(dU_temps[ns-int(dT/ t_step_measure)])
                        sample.append(I1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_temps[ns]-Q_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_leak_temps[ns]-Q_leak_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Y)

                        C_rate = float(numberlist[number][3])

                        if ((array_data_int[nt, parameters+1] < array_data_int[nt-1, parameters+1] * isc_threshold)) and (isc_threshold > 0) and isc_start == 0:
                            isc_start = 1
                        if (nt > isc_time) and (isc_time > 0) and isc_start == 0:
                            isc_start = 1
                        if  isc_start == 1:
                            writer.writerow(sample)
                            sample_N = sample_N + 1
                            samples_N = samples_N + 1

                    nt = nt + 1

            finished = (number-number_start+1)/(numberlist.shape[0]-number_start)
            time_used = (time.time() - t0)/60
            time_left =time_used / finished * (1-finished)

            process_bar(time_used,time_left, finished, '|', '100%', 50)
            print(" |P=",numberlist[number], end='', flush=True)
            # print(" |P=",numberlist[number], "|Y=",Y,"|N=",sample_N, end='', flush=True)


    samplefile_N.close()
    print(" |finished,", "sample number:", samples_N)
    return 0


def generate_V2(model="model2",
                  input_address="",
                  output_address="",
                  groupname="C",
                  filename="",
                  parameters=3,
                  capacity_period=600 ,
                  t_points=None,
                  t_step_measure=10,
                  t_step_samples=None,
                  isc_threshold = 0.9,
                  isc_time = 200,
                  time=None,
                  error_current = 0,
                  error_voltage = 0,
                  smooth_order = 3,
                  window_size=20,
                  number_start = 432,
                  period_normalized = True):

    t0 = time.time()
    samplefile_list = open(input_address+groupname+"_filelist.csv", "r")
    readerlist = csv.reader(samplefile_list)
    numberlist = np.array(list(readerlist))

    # t_step_period = t_points[1]-t_points[0]

    samples_N = 0
    samplefile_N = open(output_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order) +".csv", "w", encoding='utf-8', newline='')

    # number_start=431

    for number in range(numberlist.shape[0]):

        if (number> number_start-1):

            array_data_int=np.loadtxt(open(input_address+groupname +str(number+1) + ".csv","rb"),delimiter=",",skiprows=0)

            if len(array_data_int)>capacity_period:

                t_step_measure = int(array_data_int[1][parameters]-array_data_int[0][parameters])

                T_c = 200
                T_max = 0
                Y = 0
                window_size_r = int(window_size)
                t_period=int(capacity_period)

                if groupname == "C" or groupname == "EC" or groupname == "CT":
                    if int(numberlist[number][2])<10000:
                        Y = 1

                if groupname == "I" or groupname == "EI" or groupname == "EI1" or groupname == "IT":
                    Y = 2
                    T_max = max(array_data_int[:, parameters+3])
                    if T_max> T_c:
                        Y = 3

                # nt = round(min(array_data_int[:, 0]))+round((t_period+t_start)/t_step_measure)+smooth_order*2
                nt = 0

                writer = csv.writer(samplefile_N)

                sample_N = 0

                U_temps=[]
                I_temps=[]
                U1_temps=[]
                I1_temps=[]
                dU_temps=[]
                dI_temps=[]
                Q_temps=[0]
                Q_leak_temps=[0]
                Q_temp=0

                m_points= np.arange(0,int(window_size_r)*t_step_measure,t_step_measure)
                isc_start = 0

                while (nt < len(array_data_int[:, 0])):

                    U_temps.append(array_data_int[nt, parameters+1] + np.random.normal(0, error_voltage/3))
                    I_temps.append(array_data_int[nt, parameters+2] + np.random.normal(0, error_current/3))
                    Q_temps.append(float(I_temps[-1])*t_step_measure+Q_temps[-1])
                    Q_leak_temps.append(float(U_temps[-1])*t_step_measure+Q_leak_temps[-1])

                    if len(I_temps)> window_size_r and smooth_order > 0:
                        U_temps_m=[]
                        I_temps_m=[]
                        nm= len(I_temps)-1

                        for m in range(window_size_r):
                            U_temps_m.append(float(U_temps[nm-m]))
                            I_temps_m.append(float(I_temps[nm-m]))

                        coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
                        coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

                        dU_temps.append(coeff1[-2])
                        U1_temps.append(coeff1[-1])
                        dI_temps.append(coeff2[-2])
                        I1_temps.append(coeff2[-1])

                    write_point=nt % int(t_step_samples[Y]/t_step_measure)

                    if len(I1_temps)> round((t_period)/t_step_measure)+1 and ((model == 'model2') or ((model == 'model1') and (Y<2))) and write_point==0:
                        sample = []
                        for p in range(parameters):
                            sample.append(numberlist[number][p+1])
                        ns= len(I1_temps)-1
                        dT= t_period
                        sample.append(U1_temps[ns])
                        sample.append(dU_temps[ns])
                        sample.append(I1_temps[ns])
                        sample.append(U1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(dU_temps[ns-int(dT/ t_step_measure)])
                        sample.append(I1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_temps[ns]-Q_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_leak_temps[ns]-Q_leak_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Y)

                        C_rate = float(numberlist[number][3])
                        if ((array_data_int[nt, parameters+1]<4.34)
                                and (array_data_int[nt - int(dT / t_step_measure)- window_size_r, parameters+1]< 4.34)
                                and (abs(array_data_int[nt, parameters+2])> C_rate*3.577*0.9)
                                and (abs(array_data_int[nt - int(dT / t_step_measure) - window_size_r, parameters+2]) > C_rate*3.577*0.9)
                                and ((array_data_int[nt - int(dT / t_step_measure) - window_size_r, parameters+2]) * (array_data_int[nt ,parameters+2]) > 0)):
                            writer.writerow(sample)
                            sample_N = sample_N + 1
                            samples_N = samples_N + 1

                    if len(I1_temps)> round((t_period)/t_step_measure)+1  and (model == 'model1') and (Y>=2) and write_point==0:
                        sample = []
                        for p in range(parameters):
                            sample.append(numberlist[number][p+1])
                        ns= len(I1_temps)-1
                        dT= t_period
                        sample.append(U1_temps[ns])
                        sample.append(dU_temps[ns])
                        sample.append(I1_temps[ns])
                        sample.append(U1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(dU_temps[ns-int(dT/ t_step_measure)])
                        sample.append(I1_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_temps[ns]-Q_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Q_leak_temps[ns]-Q_leak_temps[ns-int(dT/ t_step_measure)])
                        sample.append(Y)

                        C_rate = float(numberlist[number][3])

                        if ((array_data_int[nt, parameters+1] < array_data_int[nt-1, parameters+1] * isc_threshold)) and (isc_threshold > 0) and isc_start == 0:
                            isc_start = 1
                        if (nt > isc_time) and (isc_time > 0) and isc_start == 0:
                            isc_start = 1
                        if  isc_start == 1:
                            writer.writerow(sample)
                            sample_N = sample_N + 1
                            samples_N = samples_N + 1

                    nt = nt + 1

            finished = (number-number_start+1)/(numberlist.shape[0]-number_start)
            time_used = (time.time() - t0)/60
            time_left =time_used / finished * (1-finished)

            process_bar(time_used,time_left, finished, '|', '100%', 50)
            print(" |P=",numberlist[number], end='', flush=True)
            # print(" |P=",numberlist[number], "|Y=",Y,"|N=",sample_N, end='', flush=True)


    samplefile_N.close()
    print(" |finished,", "sample number:", samples_N)
    return 0


def ifelserole(x ,y):
    group=[min(y), max(y)]
    num_feature=len(x[0])
    num_sample=len(y)
    score=0
    index=0
    xc=0

    m00=0
    m10=0
    m01=0
    m11=0

    for i in range(num_feature):

        rank = np.argsort(-x[:,i])

        num0=0
        num1=0

        for j in range(num_sample):
            if y[rank[j]]==int(group[0]):
                num0=num0+1
            if y[rank[j]]==int(group[1]):
                num1=num1+1

        num00=0
        num01=0
        num10=0
        num11=0

        for j in range(num_sample):
            # xc=x[rank[j]][i]
            if y[rank[j]]==int(group[0]):
                num00=num00+1
            if y[rank[j]]==int(group[1]):
                num01=num01+1

            num10=num0-num00
            num11=num1-num01

            # TPR=float(num00)/float(num0)
            # FPR=float(num10)/float(num0)
            # FNR=float(num01)/float(num1)
            # TNR=float(num11)/float(num1)

            # s1= (TPR+TNR)/2
            # s2= (FPR+FNR)/2
            # s1= TPR/(TPR + (FPR+FNR)/2)
            # s2= FNR/(FNR + (TNR+TPR)/2)

            s1= 0
            s2= 0
            if num00+num01!=0 and num10+num11!=0:
                s1= num00/(num00+num10)/2+num11/(num01+num11)/2

            if score< max([s1, s2]):
                score=max([s1, s2])
                index=i
                xc=x[rank[j]][i]
                m00=num00
                m10=num10
                m01=num01
                m11=num11

    return score #, index, xc, [[m00,m10], [m01,m11]]

def generate_V3(model="c1",
                input_address="",
                output_address="",
                groupname="C",
                filename="",
                parameters=3,
                capacity_period=600 ,
                t_points=None,
                t_step_measure=10,
                t_step_samples=None,
                isc_threshold = 0.9,
                isc_time = 200,
                time=None,
                error_current = 0,
                error_voltage = 0,
                smooth_order = 1,
                window_size=3,
                number_start = 432,
                period_normalized = True):

    t0 = time.time()
    samplefile_list = open(input_address+groupname+"_filelist.csv", "r")
    readerlist = csv.reader(samplefile_list)
    numberlist = np.array(list(readerlist))

    # t_step_period = t_points[1]-t_points[0]

    samples_N = 0
    samplefile_N = open(output_address+filename+"_samples_"+ groupname+ "_" + str(int(capacity_period)) +"_" + str(error_current)+ "_" + str(error_voltage) + "_" + str(smooth_order) +".csv", "w", encoding='utf-8', newline='')

    writer = csv.writer(samplefile_N)
    columns=['capacity_ratio','isc_resistance','c_rate','soc','x0','x1','x2','x3','x4','x5','x6','x7','y']
    writer.writerow(columns)

    # number_start=431
    # df = pd.DataFrame([[0,0,0,0,0,0]], columns=['capacity_ratio','isc_resistance','c_rate','soc','x','y'])

    for number in range(numberlist.shape[0]):

        if (number> number_start-1):

            array_data_int=np.loadtxt(open(input_address+groupname +str(number+1) + ".csv","rb"),delimiter=",",skiprows=0)

            if len(array_data_int)>capacity_period:

                t_step_measure = int(array_data_int[1][parameters]-array_data_int[0][parameters])

                T_c = 200
                T_max = 0
                Y = 0
                window_size_r = int(window_size)
                t_period=int(capacity_period)

                if model == "c1":
                    if int(numberlist[number][2])<10000:
                        Y = 1

                if model == "c2":
                    Y = 2
                    T_max = max(array_data_int[:, parameters+3])
                    if T_max> T_c:
                        Y = 3

                # nt = round(min(array_data_int[:, 0]))+round((t_period+t_start)/t_step_measure)+smooth_order*2
                nt = 0


                sample_N = 0

                U_temps=[]
                I_temps=[]
                U1_temps=[]
                I1_temps=[]
                dU_temps=[]
                dI_temps=[]
                Q_temps=[0]
                Q_leak_temps=[0]
                Q_temp=0

                m_points= np.arange(0,int(window_size_r)*t_step_measure,t_step_measure)
                isc_start = 0

                while (nt < len(array_data_int[:, 0])):

                    U_temps.append(array_data_int[nt, parameters+1] + np.random.normal(0, error_voltage/3))
                    I_temps.append(array_data_int[nt, parameters+2] + np.random.normal(0, error_current/3))
                    Q_temps.append(float(I_temps[-1])*t_step_measure+Q_temps[-1])
                    Q_leak_temps.append(float(U_temps[-1])*t_step_measure+Q_leak_temps[-1])

                    if len(I_temps)> window_size_r and smooth_order > 0:
                        U_temps_m=[]
                        I_temps_m=[]
                        nm= len(I_temps)-1

                        for m in range(window_size_r):
                            U_temps_m.append(float(U_temps[nm-m]))
                            I_temps_m.append(float(I_temps[nm-m]))

                        coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
                        coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

                        dU_temps.append(coeff1[-2])
                        U1_temps.append(coeff1[-1])
                        dI_temps.append(coeff2[-2])
                        I1_temps.append(coeff2[-1])

                    write_point=nt % int(t_step_samples[Y]/t_step_measure)

                    if len(I1_temps)> round((t_period)/t_step_measure)+1 and ((model == 'c1') and (Y<2)) and write_point==0:
                        sample = []
                        for p in range(parameters):
                            sample.append(numberlist[number][p+1])
                        ns= len(I1_temps)-1
                        dT= t_period
                        sample.append(U1_temps[-1]) #test
                        sample.append(dU_temps[-1])
                        sample.append(I_temps[-1])
                        sample.append(U1_temps[-1-int(dT/ t_step_measure)])
                        sample.append(dU_temps[-1-int(dT/ t_step_measure)])
                        sample.append(I_temps[-1-int(dT/ t_step_measure)])
                        sample.append(Q_temps[-1]-Q_temps[-1-int(dT/ t_step_measure)])
                        sample.append(Q_leak_temps[-1]-Q_leak_temps[-1-int(dT/ t_step_measure)])
                        sample.append(Y)

                        C_rate = float(numberlist[number][3])
                        if ((array_data_int[nt, parameters+1]<4.34)
                                and (array_data_int[nt - int(dT / t_step_measure)- window_size_r, parameters+1]< 4.34)
                                and (abs(array_data_int[nt, parameters+2])> C_rate*3.577*0.9)
                                and (abs(array_data_int[nt - int(dT / t_step_measure) - window_size_r, parameters+2]) > C_rate*3.577*0.9)
                                and ((array_data_int[nt - int(dT / t_step_measure) - window_size_r, parameters+2]) * (array_data_int[nt ,parameters+2]) > 0)):
                            columns=list(numberlist[number][1:4])
                            columns.extend([0.0])
                            columns.extend(list(sample[-9:-1]))
                            columns.extend([sample[-1]])
                            writer.writerow(columns)
                            # writer.writerow(sample)
                            # df2 = pd.DataFrame([[numberlist[number][1],numberlist[number][2],numberlist[number][3],0.0,sample[-9:-1],sample[-1]]], columns=['capacity_ratio','isc_resistance','c_rate','soc','x','y'])
                            # # print(df2)
                            # df=pd.concat([df,df2], ignore_index=True)
                            # print(df)

                            sample_N = sample_N + 1
                            samples_N = samples_N + 1

                    if len(I1_temps)> round((t_period)/t_step_measure)+1  and (model == 'c2') and (Y>=2) and write_point==0:
                        sample = []
                        for p in range(parameters):
                            sample.append(numberlist[number][p+1])
                        ns= len(I1_temps)-1
                        dT= t_period
                        sample.append(U_temps[-1])
                        sample.append(dU_temps[-1])
                        sample.append(I_temps[-1])
                        sample.append(U_temps[-1-int(dT/ t_step_measure)])
                        sample.append(dU_temps[-1-int(dT/ t_step_measure)])
                        sample.append(I_temps[-1-int(dT/ t_step_measure)])
                        sample.append(Q_temps[-1]-Q_temps[-1-int(dT/ t_step_measure)])
                        sample.append(Q_leak_temps[-1]-Q_leak_temps[-1-int(dT/ t_step_measure)])
                        sample.append(Y)

                        C_rate = float(numberlist[number][3])

                        if ((array_data_int[nt, parameters+1] < array_data_int[nt-1, parameters+1] * isc_threshold)) and (isc_threshold > 0) and isc_start == 0:
                            isc_start = 1
                        if (nt > isc_time) and (isc_time > 0) and isc_start == 0:
                            isc_start = 1
                        if  isc_start == 1:
                            columns=list(numberlist[number][1:5])
                            columns.extend(list(sample[-9:-1]))
                            columns.extend([sample[-1]])
                            writer.writerow(columns)
                            sample_N = sample_N + 1
                            samples_N = samples_N + 1

                    nt = nt + 1

            finished = (number-number_start+1)/(numberlist.shape[0]-number_start)
            time_used = (time.time() - t0)/60
            time_left =time_used / finished * (1-finished)

            process_bar(time_used,time_left, finished, '|', '100%', 50)
            print(" |P=",numberlist[number], end='', flush=True)
            # print(" |P=",numberlist[number], "|Y=",Y,"|N=",sample_N, end='', flush=True)


    samplefile_N.close()
    # df.to_csv('df_test.csv',index=False)
    print(" |finished,", "sample number:", samples_N)
    return 0



