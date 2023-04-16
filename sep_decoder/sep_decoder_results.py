import numpy as np
import sys 
import os
from sep_decoder_proj import sep_dynamics_2

L = 10
p = 0.0
number_shots = 2000 
number_circuit_realis = 10

accuracy_array = []
for circuit_iter in range(1,number_circuit_realis+1):
    print(circuit_iter)
    try: 
        measurement_record_0 = np.load("learnability_transitions_cluster/data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,0,number_shots,circuit_iter))        
        measurement_record_1 = np.load("learnability_transitions_cluster/data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,1,number_shots,circuit_iter))        
        measurement_records = np.concatenate([measurement_record_0,measurement_record_1],axis=0)
        num_meas_records_0 = len(measurement_record_0[:,0,0])
        num_meas_records_1 = len(measurement_record_1[:,0,0])  
        num_meas_records = num_meas_records_0+num_meas_records_1
        permut = np.random.permutation(num_meas_records) 
        data = measurement_records[permut,:,:]
        charge_output_0 = np.zeros(num_meas_records_0)
        charge_output_1 = np.ones(num_meas_records_1)
        charge_output = np.concatenate([charge_output_0,charge_output_1],axis=0)
        labels = charge_output[permut]
        test_percentage = 0.2 
        train_percentage = 1 - test_percentage 
        number_samples = len(measurement_records)
        test_data_number_samples = round(test_percentage * number_samples)
        test_data = data[0:test_data_number_samples,:,:]
        num_different_records = len(measurement_record_0[:,0,0])
        accuracy = []
        for i in range(0,test_data_number_samples):
            charge = int(L//2 + labels[i])
            proba_success = sep_dynamics_2(test_data[i,:,:], charge)
            accuracy.append(proba_success > 0.5)
        accuracy_array.append(np.mean(accuracy))
        print("acc", np.mean(accuracy))
    except: 
        print("ignore circuit iter ", circuit_iter)    
#np.save("learnability_transitions_cluster/data/test_accuracy_SEP_L_{}_p_{}_numbershots_{}.npy".format(L,p,number_shots), accuracy_array) # store all measurements except for final measurements 