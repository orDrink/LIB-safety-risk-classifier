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

def datascaler_V3(X=None, X_max=None, X_min=None, X_mean=None):
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            X[i][j]=(X[i][j]-X_mean[j])/(X_max[j]-X_min[j])
    return X

def process_bar(time_used, time_left, percent, start_str='|', end_str='100%', total_length=50):
    bar = ''.join(['▋'] * int(percent * total_length)) + ''
    # "\033[31m%s\033[0m"
    bar = '\r' + start_str + bar.ljust(total_length) + '|{:0>4.1f}%|'.format(percent*100) + end_str
    print(bar," |used",'%.2f' % time_used,"mins |left", '%.2f' % time_left,"mins", end='', flush=True)


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



