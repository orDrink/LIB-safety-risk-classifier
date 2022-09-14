import math
import csv
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
import time
import random
import function

x_num_feature = 10
t_step_periods = [300]


errors = [[0, 0],[0, 1e-5],[0, 1e-4],[0, 1e-3],[0, 1e-2],[0, 1e-1]]
# errors = [[0, 0],[0, 1e-5],[0, 1e-4],[0, 1e-3],[0, 1e-2],[0, 1e-1],[1e-5, 0],[1e-4, 0],[1e-3, 0],[1e-2, 0],[1e-1, 0],[1, 0]]

address1 ="D:/ml safety risk/Project/Splited_data/"
address2 ="D:/ml safety risk/Project/Training_data/"
address3 ="D:/ml safety risk/Project/Original_data/"


I_data_file=["charging ISC data V2", "discharging ISC data V2"]
I_data_file=["charging ISC data V2-TEST"]

function.divide(input_address=address3,
                output_address=address1,
                groupname="IT",
                parameters=4,
                datasets=I_data_file)

CT_data_file=["cycling data for test V1"]
function.divide(input_address=address3,
                output_address=address1,
                groupname="CT",
                parameters=2,
                datasets=CT_data_file)


t_start = 1
t_ISC = 200

t_step_measure = 10

input_address ="D:/ml safety risk/Project/Splited_data/"
output_address = "D:/ml safety risk/Project/Training_data/"

for error in errors:
    print('')

    capacity_periods =  [120,240,600,900]
    for capacity_period in capacity_periods:
    t_step_samples = [10, 10, 1, 1]
    function.generate_V3(model="c1",
                           input_address=input_address,
                           output_address=output_address,
                           filename="df",
                           groupname="C",
                           parameters=3,
                           capacity_period = capacity_period,
                           t_step_measure = 10,
                           t_step_samples = t_step_samples,
                           time = time,
                           error_current = error[0],
                           error_voltage = error[1],
                           smooth_order = 1,
                           window_size= 3)
    
    t_step_samples = [10, 10, 1, 1]
    function.generate_V3(model="c2",
                           input_address=input_address,
                           output_address=output_address,
                           filename="df",
                           groupname="I",
                           parameters=4,
                           capacity_period = 60,
                           t_step_measure = 1,
                           t_step_samples = t_step_samples,
                           isc_threshold = 0,
                           isc_time = 200,
                           time =time,
                           error_current = error[0],
                           error_voltage = error[1],
                           smooth_order = 1,
                           window_size = 3,
                           number_start = 0)
