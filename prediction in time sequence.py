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

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib

from scipy.signal import butter, lfilter, freqz

C0= 'c0_dt_scaled_test_V3_ei0_eu0_t60.model'
C1= 'svc_scaled_test_V3_HPC_r20_500_ei0_eu0_t600.model'
C2= 'c2_dt_scaled_test_V3_ei0_eu0_t60.model'

classifer_c0 = joblib.load(C0)
classifer_c1 = joblib.load(C1)
classifer_c2 = joblib.load(C2)

input_address="D:/ml safety risk/Project/Splited_data/"
filename="C435" # C435 #C467 I80 CT2 CT8 IT1/2

data=np.loadtxt(open(input_address+filename + ".csv","rb"),delimiter=",",skiprows=0)
parameters=3

if filename[0]=='C':
    parameters=len(data[0])-3
elif filename[0]=='I':
    parameters=len(data[0])-4

t_step= data[1, parameters]-data[0, parameters]
frequency = 1/t_step  #input frequency

print(parameters, t_step, frequency)

smooth_window = 3
smooth_order = 1

period_short = 60
period_long= 600

num_feature= 8
current_1C= 3.577

X_long_max = [4.339977261, 0.007368864, 7.154, 4.323243104, 0.003604036, 7.154, 4292.4, 2599.034436]
X_long_min = [2.790661502, -0.000782145, -7.154, 3.139090545, -0.00232158, -7.154, -4292.4, 1798.51833]
X_long_mean = [3.7538, 0.0000, 0.1184, 3.7576, 0.0000, 0.1184, 71.0692, 2253.8703]

X_short_max = [4.339977261, 2.191329391, 7.154, 4.382654491, 2.191329391, 7.154, 4292.4,2599.034436 ]
X_short_min = [-0.000419, -0.23662166, 0, -0.000419, -0.23662166, -7.154, -414.932 ,-0.0015552]
X_short_mean = [3.595421184, 0.00109842, 0.099146893, 3.664921519, 0.001063486, 0.073874705, 58.77931943, 1916.457517]

X=[0]*num_feature

dU=[]
U=[]
I=[]
Qc=[0]
Uc=[0]

smooth_window_long = int(smooth_window)
sequence_len_short = int(period_short*frequency)
sequence_len_long = round(period_long*frequency)
m_points= np.arange(0,int(smooth_window_long)*t_step,t_step)

U_temps=[]
I_temps=[]
U1_temps=[]
I1_temps=[]
dU_temps=[]
dI_temps=[]
Q_temps=[0]
Q_leak_temps=[0]
Q_temp=0
x=[]
y=[]
z=[]

step = 0
for step in range(len(data)):

    risk = -1
    U_temps.append(data[step, parameters+1])
    I_temps.append(data[step, parameters+2])
    Q_temps.append(float(I_temps[-1])*t_step+Q_temps[-1])
    Q_leak_temps.append(float(U_temps[-1])*t_step+Q_leak_temps[-1])

    if len(I_temps)> smooth_window_long and smooth_order > 0:
        U_temps_m=[]
        I_temps_m=[]
        nm= len(I_temps)-1

        for m in range(smooth_window_long):
            U_temps_m.append(float(U_temps[nm-m]))
            I_temps_m.append(float(I_temps[nm-m]))

        coeff1 = np.polyfit(np.array(m_points), np.array(U_temps_m), smooth_order)
        coeff2 = np.polyfit(np.array(m_points), np.array(I_temps_m), smooth_order)

        dU_temps.append(coeff1[-2])
        U1_temps.append(coeff1[-1])
        dI_temps.append(coeff2[-2])
        I1_temps.append(coeff2[-1])

    if step>smooth_window_long+sequence_len_short:
        if ((data[step, parameters+1]<4.34)
                # and (abs(data[step, parameters+2])> 3.577*0.01)
                # and (abs(data[step - sequence_len_short- smooth_window_long, parameters+2]) > 3.577*0.01)
                and (data[step - sequence_len_short- smooth_window_long, parameters+1]< 4.34)):
            sample = []
            ns= len(I1_temps)-1
            sample.append(U1_temps[-1])
            sample.append(dU_temps[-1])
            sample.append(I1_temps[-1])
            sample.append(U1_temps[-1-sequence_len_short])
            sample.append(dU_temps[-1-sequence_len_short])
            sample.append(I1_temps[-1-sequence_len_short])
            sample.append(Q_temps[-1]-Q_temps[-1-sequence_len_short])
            sample.append(Q_leak_temps[-1]-Q_leak_temps[-1-sequence_len_short])
            X_short =[np.array(sample)]
            X_short =function.datascaler_V3(X=X_short, X_max=X_short_max, X_min=X_short_min, X_mean=X_short_mean)
            # print(X_short)
            output0 = classifer_c0.predict(X_short)
            if (output0[-1]==2):
                risk = classifer_c2.predict(X_short)[-1]
            else:
                risk = 0

    if step>smooth_window_long+sequence_len_long:
        # C_rate=data[step][parameters-1]
        if ((data[step, parameters+1]<4.34)
                and (data[step - sequence_len_long- smooth_window_long, parameters+1]< 4.34)
                and (abs(data[step, parameters+2])> 3.577*0.01)
                and (abs(data[step - sequence_len_short- smooth_window_long, parameters+2]) > 3.577*0.01)
                and ((data[step - sequence_len_long- smooth_window_long, parameters+2]) * (data[step ,parameters+2]) > 0)):
            # print(X_long)
            # X_long =function.datascaler_rf(X=X_long, X_max=X_long_max, X_min=X_long_min)
            if (risk == 0):
                sample = []
                ns= len(I1_temps)-1
                sample.append(U_temps[-1])
                sample.append(dU_temps[-1])
                sample.append(I_temps[-1])
                sample.append(U_temps[-1-sequence_len_long])
                sample.append(dU_temps[-1-sequence_len_long])
                sample.append(I_temps[-1-sequence_len_long])
                sample.append(Q_temps[-1]-Q_temps[-1-sequence_len_long])
                sample.append(Q_leak_temps[-1]-Q_leak_temps[-1-sequence_len_long])

                X_long =[np.array(sample)]
                X_long=function.datascaler_V3(X=X_long, X_max=X_long_max, X_min=X_long_min, X_mean=X_long_mean)
                output1 = classifer_c1.predict(X_long)
                if (output1==1):
                    risk = 1
                else:
                    risk = 0

    x.append(step)
    y.append(data[step, parameters+1])
    z.append(risk)

x = np.array(x)
y = np.array(y)
z = np.array(z)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Setting standard filter requirements.
order = 5
fs = 1
cutoff = 0.1

z_filtered=z

for step in range(len(data)):
    print(step,data[step, parameters+1],data[step, parameters+2],z_filtered[step],z[step])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y, '-', label = 'Voltage')

ax2 = ax.twinx()
# plt.scatter(x,z,c='red',s=2)
ax2.plot(x,z_filtered, '-r', label = 'Risk level')
fig.legend(loc=1, bbox_to_anchor=(0.5,0.2), bbox_transform=ax.transAxes)

ax.set_xlabel("Time step")
ax.set_ylabel(r"Voltage (V)")
ax2.set_ylabel(r"Risk level")

plt.savefig('test.png')
plt.show()














