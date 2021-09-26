""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
import sys
import math
import LinRegLearner as lrl
import DTLearner as dl
import RTLearner as rl
import BagLearner as bl
import InsaneLearner as il
import matplotlib
import matplotlib.pyplot as plt
import csv
import time
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.genfromtxt(inf, delimiter=',')
    data = data[1:, 1:]
    data.astype(float)


    # compute how much of the data is training and testing  		  	   		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  

    #create experience 1 learner and train it
    dt_in_sample_rmse = []
    dt_out_sample_rmse = []
    leaf_range = range(1,50)

    for i in leaf_range:
        learner = dl.DTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)

        #evaluate in sample
        pred_y_1 = learner.query(train_x)
        rmse_1 = np.sqrt(np.mean((pred_y_1 - train_y)**2))
        dt_in_sample_rmse.append(rmse_1)

        #evaluate out of sample
        pred_y_2 = learner.query(test_x)
        rmse_2 = np.sqrt(np.mean((pred_y_2 - test_y)**2))
        dt_out_sample_rmse.append(rmse_2)

    #figure for experiment1
    plt.figure()
    plt.plot(leaf_range, dt_in_sample_rmse, label= 'in sample')
    plt.plot(leaf_range, dt_out_sample_rmse, label= 'out sample')
    plt.title('RMSE vs. leaf size - DTLearner')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('experiment_1')


    # create experience 2 learner and train it
    bl_in_sample_rmse = []
    bl_out_sample_rmse = []

    for i in leaf_range:
        learner = bl.BagLearner(learner=dl.DTLearner, kwargs={'leaf_size':i},
                                bags=20, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y_1 = learner.query(train_x)
        rmse_1 = np.sqrt(np.mean((pred_y_1 - train_y) ** 2))
        bl_in_sample_rmse.append(rmse_1)

        # evaluate out of sample
        pred_y_2 = learner.query(test_x)
        rmse_2 = np.sqrt(np.mean((pred_y_2 - test_y) ** 2))
        bl_out_sample_rmse.append(rmse_2)

    #figure for experiment2
    plt.figure()
    plt.plot(leaf_range, bl_in_sample_rmse, label= 'in sample')
    plt.plot(leaf_range, bl_out_sample_rmse, label= 'out sample')
    plt.title('RMSE vs. leaf size - BagLearner')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('experiment_2')


    #experiment_3_a compare length and MAE of DT and RT
    # dt_in_sample_mae = []
    # dt_out_sample_mae = []
    # rt_in_sample_mae = []
    # rt_out_sample_mae = []
    R_squared_dt_in_sample = []
    R_squared_dt_out_sample = []
    R_squared_rt_in_sample = []
    R_squared_rt_out_sample = []
    train_time_dt = []
    train_time_rt = []


    for i in leaf_range:

        learner_1 = dl.DTLearner(leaf_size=i, verbose=False)
        start_time = time.time()
        learner_1.add_evidence(train_x, train_y)
        end_time = time.time()
        train_time_dt.append(end_time - start_time)
        pred_y_1 = learner_1.query(train_x)
        # mae_1 = np.mean(abs(pred_y_1 - train_y))
        # dt_in_sample_mae.append(mae_1)
        R_squared_1 = 1 - np.sum((pred_y_1 - train_y)**2)/np.sum((train_y - np.mean(train_y))**2)
        R_squared_dt_in_sample.append(R_squared_1)

        pred_y_2 = learner_1.query(test_x)
        # mae_2 = np.mean(abs(pred_y_2 - test_y))
        # dt_out_sample_mae.append(mae_2)
        R_squared_2 = 1 - np.sum((pred_y_2 - test_y) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2)
        R_squared_dt_out_sample.append(R_squared_2)

        learner_2 = rl.RTLearner(leaf_size=i, verbose=False)
        start_time = time.time()
        learner_2.add_evidence(train_x, train_y)
        end_time = time.time()
        train_time_rt.append(end_time - start_time)
        pred_y_3 = learner_2.query(train_x)
        # mae_3 = np.mean(abs(pred_y_3 - train_y))
        # rt_in_sample_mae.append(mae_3)
        R_squared_3 = 1 - np.sum((pred_y_3 - train_y) ** 2) / np.sum((train_y - np.mean(train_y)) ** 2)
        R_squared_rt_in_sample.append(R_squared_3)

        pred_y_4 = learner_2.query(test_x)
        # mae_4 = np.mean(abs(pred_y_4 - test_y))
        # rt_out_sample_mae.append(mae_4)
        R_squared_4 = 1 - np.sum((pred_y_4 - test_y) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2)
        R_squared_rt_out_sample.append(R_squared_4)

    # plt.figure()
    # plt.plot(leaf_range, dt_in_sample_mae, label='DTLearner MAE in sample')
    # plt.plot(leaf_range, rt_in_sample_mae, label='RTLearner MAE in sample')
    # plt.plot(leaf_range, dt_out_sample_mae, label='DTLearner MAE out sample')
    # plt.plot(leaf_range, rt_out_sample_mae, label='RTLearner MAE out sample')
    # plt.title(' MAE vs. leaf size - DTLearner and RTLearner ')
    # plt.xlabel('leaf size')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.savefig('experiment_3_1')

    plt.figure()
    plt.plot(leaf_range, train_time_dt, label='DTLearner train time ')
    plt.plot(leaf_range, train_time_rt, label='RTLearner train time ')
    plt.title(' train time vs. leaf size - DTLearner and RTLearner ')
    plt.xlabel('leaf size')
    plt.ylabel('train time')
    plt.legend()
    plt.savefig('experiment_3_1')

    plt.figure()
    plt.plot(leaf_range, R_squared_dt_in_sample, label='DTLearner R-Squared in sample')
    plt.plot(leaf_range, R_squared_dt_out_sample, label='DTLearner R-Squared out sample')
    plt.plot(leaf_range, R_squared_rt_in_sample, label='RTLearner R-Squared in sample')
    plt.plot(leaf_range, R_squared_rt_out_sample, label='RTLearner R-Squared out sample')
    plt.title(' R-Squared vs. leaf size - DTLearner and RTLearner ')
    plt.xlabel('leaf size')
    plt.ylabel('R-Squared')
    plt.legend()
    plt.savefig('experiment_3_2')

    # plt.figure()
    # plt.plot(leaf_range, bl_in_sample_rmse, label= 'in sample')
    # plt.plot(leaf_range, bl_out_sample_rmse, label= 'out sample')
    # plt.title('RMSE vs. leaf size - BagLearner')
    # plt.xlabel('leaf size')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.savefig('experiment_2')





