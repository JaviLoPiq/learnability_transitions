import time
import numpy as np
import pickle
from itertools import combinations
import qiskit as qk
import sys
import h5py 
import random
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC, dict_to_array_measurements
from quantum_decoder_proj import quantum_dynamics_2
L = 10
depth = L
p = 0.2
number_shots = 2000 
number_circuit_realis = 1
depth_ratio = 1
scrambling_type = 'Special'
is_noisy = False
decoding_protocol = 3

np.random.seed(1)
accuracy_array = []
for circuit_iter in range(1,number_circuit_realis+1):
    print(circuit_iter)
    #try:   
    measurement_record_0 = dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, 0)     
    measurement_record_1 = dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, 1) 
    print(np.shape(measurement_record_0))
    with h5py.File('learnability_transitions_cluster/data/initial_state_L_{}_p_{}_Q_{}_iter_{}.hdf5'.format(L, p, 5, circuit_iter), 'r') as f:
        # Retrieve the dataset and convert it to a numpy array
        my_array = f['statevector'][()]
        state = np.array(my_array.tolist())
        initial_state_Q = state.reshape((2,)*L)
    with h5py.File('learnability_transitions_cluster/data/initial_state_L_{}_p_{}_Q_{}_iter_{}.hdf5'.format(L, p, 6, circuit_iter), 'r') as f:
        # Retrieve the dataset and convert it to a numpy array
        my_array = f['statevector'][()]
        state = np.array(my_array.tolist())
        initial_state_Q2 = state.reshape((2,)*L)    
    #TODO: replace unitaries by its parameters?
    U_list = np.load('learnability_transitions_cluster/data/unitaries_L_{}_p_{}_Q_{}_iter_{}.npy'.format(L, p, 0, circuit_iter), allow_pickle=True)
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
    for i in range(0,1):
        charge = int(L//2 + labels[i])
        if labels[i] == 0:
            print(quantum_dynamics_2(data[i,:,:], charge, U_list, initial_state_Q, initial_state_Q2, decoding_protocol=decoding_protocol))
            accuracy.append(quantum_dynamics_2(data[i,:,:], charge, U_list, initial_state_Q, initial_state_Q2, decoding_protocol=decoding_protocol)[depth-2] > 0.5) 
        else:
            print(quantum_dynamics_2(data[i,:,:], charge, U_list, initial_state_Q2, initial_state_Q, decoding_protocol=decoding_protocol))
            accuracy.append(quantum_dynamics_2(data[i,:,:], charge, U_list, initial_state_Q2, initial_state_Q, decoding_protocol=decoding_protocol)[depth-2] > 0.5) 
        #else:
            #accuracy.append(quantum_dynamics_2(test_data[i,:,:], charge, U_list, scrambled_state_1, decoding_protocol=decoding_protocol)[2*(depth-1)] > 0.5) 
    accuracy_array.append(np.mean(accuracy)) 
    print("acc", np.mean(accuracy))  
    """
    except: 
        print("ignore circuit iter ", circuit_iter)  
    """