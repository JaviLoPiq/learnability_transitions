import numpy as np
import sys 
import os
from sep_decoder_weak import sep_dynamics_2
#from U1MRC import dict_to_array_measurements
#sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
import pickle
import h5py

L = 6
depth = L//2
np.random.seed(1) # fix seed so that all decoders "decode" same training data
accuracy_array = []
p_list = list(np.round(np.linspace(0.2*np.pi/2,0.8*np.pi/2,15),3))
p_list.extend(list(np.array([0.001,0.01,0.02,0.05,0.1,0.15])*np.pi/2))
gamma_arr = [p_list[10]]
num_shots = 10000
np.random.seed(3)
for gamma in gamma_arr:
    with open('Weak measurements/data/qiskit_data/measurement_data_all_qubits_special_depth_ratio=0.5/L={}_depth={}_Q={}_p={}_shots={}_seed=1.imdat'.format(L,depth,L//2,gamma,num_shots), 'rb') as f:
        data_raw,_,_ = pickle.load(f) # contains repeated samples
    
    measurement_record_0 = np.zeros((num_shots,depth-1,L))
    index_sample = 0
    for data_sample in data_raw: # convert data to np array
        for i in range(data_sample[1]):
            measurement_record_0[index_sample,:,:] = data_sample[0][0:depth-1,:]
            index_sample += 1         
    with open('Weak measurements/data/qiskit_data/measurement_data_all_qubits_special_depth_ratio=0.5/L={}_depth={}_Q={}_p={}_shots={}_seed=1.imdat'.format(L,depth,L//2-1,gamma,num_shots), 'rb') as f:
        data_raw,_,_ = pickle.load(f)
    
    measurement_record_1 = np.zeros((num_shots,depth-1,L))
    index_sample = 0
    for data_sample in data_raw: # convert data to np array
        for i in range(data_sample[1]):
            measurement_record_1[index_sample,:,:] = data_sample[0][0:depth-1,:]
            index_sample += 1
    print(index_sample)
    measurement_records = np.concatenate([measurement_record_0,measurement_record_1],axis=0)
    num_meas_records_0 = len(measurement_record_0[:,0,0])
    num_meas_records_1 = len(measurement_record_1[:,0,0])   
    num_meas_records = num_meas_records_0+num_meas_records_1
    charge_output_0 = np.zeros(num_meas_records_0)
    charge_output_1 = np.ones(num_meas_records_1)
    charge_output = np.concatenate([charge_output_0,charge_output_1],axis=0)
    permut = np.random.permutation(num_meas_records) 
    data = measurement_records[permut,:,:]
    labels = charge_output[permut]            
    accuracy = []
    array_lists = []
    proba_success_lists = []
    for i in range(0,num_meas_records):
        charge = int(L//2 - labels[i])
        proba_success = sep_dynamics_2(data[i,:,:], charge, gamma)
        accuracy.append(proba_success > 0.5) 
    print("acc ", np.mean(accuracy))        
    #np.save("learnability_transitions_cluster/data/accuracy_weak_SEP_decoder_L_{}_p_{}_number_shots_{}.npy".format(L,gamma,num_shots), np.mean(accuracy)) # store all measurements except for final measurements 