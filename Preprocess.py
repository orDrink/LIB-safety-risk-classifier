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


# errors = [[0, 0.0001],[0, 0.001]]
errors = [[0, 0],[0, 1e-5],[0, 1e-4],[0, 1e-3],[0, 1e-2],[0, 1e-1]]
# errors = [[0, 0],[0, 1e-5],[0, 1e-4],[0, 1e-3],[0, 1e-2],[0, 1e-1],[1e-5, 0],[1e-4, 0],[1e-3, 0],[1e-2, 0],[1e-1, 0],[1, 0]]
# errors = [[1e-1, 0],[1, 0]]
# errors = [[0, 0]]
# error_current = 0

# C_data_file=["cycling_data_1", "cycling_data_2"]
# divide("C", C_data_file, 3)
address1 ="D:/ml safety risk/Project/Splited_data/"
address2 ="D:/ml safety risk/Project/Training_data/"
address3 ="D:/ml safety risk/Project/Original_data/"


# I_data_file=["charging ISC data V2", "discharging ISC data V2"]
# I_data_file=["charging ISC data V2-TEST"]
# function.divide(input_address=address3,
#                 output_address=address1,
#                 groupname="IT",
#                 parameters=4,
#                 datasets=I_data_file)

# CT_data_file=["cycling data for test V1"]
# function.divide(input_address=address3,
#                 output_address=address1,
#                 groupname="CT",
#                 parameters=2,
#                 datasets=CT_data_file)

# f_step = 60
# t_step = int(t_period / f_step)
t_start = 1
t_ISC = 200
#
t_step_measure = 10

input_address ="D:/ml safety risk/Project/Splited_data/"
output_address = "D:/ml safety risk/Project/Training_data/"

# capacity_periods = [60]


for error in errors:
    print('')

    capacity_periods =  [120,240,600,900]
    for capacity_period in capacity_periods:
    # t_points = np.arange(0,int(t_step_period),int(t_step_period/x_num_feature))
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
    #
    # t_step_samples = [10, 10, 1, 1]
    # function.generate_V3(model="c2",
    #                         input_address=input_address,
    #                        output_address=output_address,
    #                        filename="df",
    #                        groupname="I",
    #                        parameters=4,
    #                        capacity_period = 60,
    #                        t_step_measure = 1,
    #                        t_step_samples = t_step_samples,
    #                        isc_threshold = 0,
    #                        isc_time = 200,
    #                        time =time,
    #                        error_current = error[0],
    #                        error_voltage = error[1],
    #                        smooth_order = 1,
    #                        window_size = 3,
    #                        number_start = 0)

    # t_step_samples = [10, 10, 1, 1]
    # function.generate_V2(input_address=input_address,
    #                                  output_address=output_address,
    #                                  filename="dt",
    #                                  groupname="C",
    #                                  parameters=3,
    #                                  capacity_period = capacity_period,
    #                                  t_step_measure =t_step_measure,
    #                                  t_step_samples = t_step_samples,
    #                                  time = time,
    #                                  error_current = error[0],
    #                                  error_voltage = error[1],
    #                                  smooth_order = 1,
    #                                  window_size= 3)
    # t_step_samples = [10, 50, 1, 1]
    # function.generate_V2(input_address=input_address,
    #                      output_address=output_address,
    #                      filename="dt_balanced",
    #                      groupname="C",
    #                      parameters=3,
    #                      capacity_period = capacity_period,
    #                      t_step_measure =t_step_measure,
    #                      t_step_samples = t_step_samples,
    #                      time = time,
    #                      error_current = error[0],
    #                      error_voltage = error[1],
    #                      smooth_order = 1,
    #                      window_size= 3)

    # function.generate_V2(model="model1",
    #                                input_address=input_address,
    #                                output_address=output_address,
    #                                filename="dt",
    #                                groupname="I",
    #                                parameters=4,
    #                                capacity_period = capacity_period,
    #                                t_step_measure = 1,
    #                                t_step_samples = t_step_samples,
    #                                isc_threshold = 0,
    #                                isc_time = 200,
    #                                time =time,
    #                                error_current = error[0],
    #                                error_voltage = error[1],
    #                                smooth_order = 1,
    #                                window_size = 3,
    #                                number_start = 10)

    # function.generate(input_address, output_address, "C", 3, t_points, t_step_measure, t_start, t_step_samples, time, error_current = error[0], error_voltage = error[1])
    #     function.generate_Normalized(input_address=input_address,
    #                                  output_address=output_address,
    #                                  filename="rf_normalized",
    #                                  groupname="C",
    #                                  parameters=3,
    #                                  capacity_period = capacity_period,
    #                                  t_step_measure =t_step_measure,
    #                                  t_step_samples = t_step_samples,
    #                                  time =time,
    #                                  error_current = error[0],
    #                                  error_voltage = error[1],
    #                                  smooth_order = 3,
    #                                  window_size=20)

    # function.generate_Normalized_2(model="model1",
    #                                input_address=input_address,
    #                                output_address=output_address,
    #                                filename="rf_normalized_3",
    #                                groupname="I",
    #                                parameters=4,
    #                                capacity_period = capacity_period,
    #                                t_step_measure = 1,
    #                                t_step_samples = t_step_samples,
    #                                isc_threshold = 0,
    #                                isc_time = 200,
    #                                time =time,
    #                                error_current = error[0],
    #                                error_voltage = error[1],
    #                                smooth_order = 1,
    #                                window_size = 3,
    #                                number_start = 0)

    # function.generate_Normalized_2(model="model2",
    #                                input_address=input_address,
    #                                output_address=output_address,
    #                                filename="rf_normalized",
    #                                groupname="C",
    #                                parameters=3,
    #                                capacity_period = capacity_period,
    #                                t_step_measure = 10,
    #                                t_step_samples = t_step_samples,
    #                                isc_threshold = 0,
    #                                isc_time = 0,
    #                                time =time,
    #                                error_current = error[0],
    #                                error_voltage = error[1],
    #                                smooth_order = 3,
    #                                window_size = 20,
    #                                number_start = 432)
    # #
    #
    # function.generate_Normalized_2(model="model1",
    #                                input_address=input_address,
    #                                output_address=output_address,
    #                                filename="dt_4_C0",
    #                                groupname="EC",
    #                                parameters=3,
    #                                capacity_period = capacity_period,
    #                                t_step_measure = 10,
    #                                t_step_samples = t_step_samples,
    #                                isc_threshold = 0,
    #                                isc_time = 0,
    #                                time =time,
    #                                error_current = error[0],
    #                                error_voltage = error[1],
    #                                smooth_order = 1,
    #                                window_size = 5,
    #                                number_start = 0)
    #
    # function.generate_Normalized_2(model="model1",
    #                              input_address=input_address,
    #                              output_address=output_address,
    #                              filename="dt_4_C0",
    #                              groupname="EI",
    #                              parameters=4,
    #                              capacity_period = capacity_period,
    #                              t_step_measure = 1,
    #                              t_step_samples = t_step_samples,
    #                              time =time,
    #                              error_current = error[0],
    #                              error_voltage = error[1],
    #                              smooth_order = 1,
    #                              window_size = 5,
    #                              number_start = 0)
    #
    # function.generate_Normalized_2(model="model1",
    #                                input_address=input_address,
    #                                output_address=output_address,
    #                                filename="dt_4_C0",
    #                                groupname="EI1",
    #                                parameters=4,
    #                                capacity_period = capacity_period,
    #                                t_step_measure = 1,
    #                                t_step_samples = t_step_samples,
    #                                isc_threshold = 4.195/4.198,
    #                                time =time,
    #                                error_current = error[0],
    #                                error_voltage = error[1],
    #                                smooth_order = 1,
    #                                window_size = 5,
    #                                number_start = 0)

