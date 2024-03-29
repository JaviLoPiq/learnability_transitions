import time
import numpy as np
import pickle
from itertools import combinations
import qiskit as qk
import sys
import h5py 
import random
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC, dict_to_array_measurements, fixed_charge_state
from quantum_decoder_proj import quantum_dynamics_2

L = 14
depth = L
p = 0.2
number_shots = 40000
number_circuit_realis = 1
depth_ratio = 1
scrambling_type = 'Special'
is_noisy = False
save_unitaries = False 

np.random.seed(1)
accuracy_array = []
for circuit_iter in range(number_circuit_realis,number_circuit_realis+1):
    #try:   
    measurement_record_0 = dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, L//2)     
    measurement_record_1 = dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, L//2+1) 
    #TODO: replace unitaries by its parameters?
    U_list = np.load('learnability_transitions_cluster/data/unitaries_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), allow_pickle=True)
    #m_locs = np.load('learnability_transitions_cluster/data/measurement_locs_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), allow_pickle=True)   
    measurement_records = np.concatenate([measurement_record_0,measurement_record_1], axis=0)
    num_meas_records_0 = len(measurement_record_0[:,0,0])
    num_meas_records_1 = len(measurement_record_1[:,0,0])  
    num_meas_records = num_meas_records_0+num_meas_records_1
    permut = np.random.permutation(num_meas_records) 
    data = measurement_records[permut,:,:]
    charge_output_0 = np.zeros(num_meas_records_0)
    charge_output_1 = np.ones(num_meas_records_1)
    charge_output = np.concatenate([charge_output_0,charge_output_1], axis=0)
    labels = charge_output[permut]
    test_percentage = 0.2
    train_percentage = 1 - test_percentage 
    number_samples = len(measurement_records)
    test_data_number_samples = round(test_percentage * number_samples)
    num_different_records = len(measurement_record_0[:,0,0])
    accuracy = []
    for i in range(0,test_data_number_samples):
        print(i/test_data_number_samples)
        charge = int(L//2 + labels[i])               
        proba_success = quantum_dynamics_2(data[i,:,:], charge, U_list)[depth-2] # use equal superposition protocol
        accuracy.append(proba_success > 0.5) 
    accuracy_array.append(np.mean(accuracy))               
    #except: 
    print("ignore circuit iter ", circuit_iter)  
    #np.save("learnability_transitions_cluster/data/accuracy_quantum_decoder_L_{}_p_{}_circuit_iter_{}_number_shots_{}.npy".format(L,p,circuit_iter,number_shots), accuracy) # store all measurements except for final measurements 
    print("acc", np.mean(accuracy_array))      