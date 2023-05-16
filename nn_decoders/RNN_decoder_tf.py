import tensorflow as tf
from tensorflow.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #do not show tensorflow warnings
import numpy as np 
import sys
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC, dict_to_array_measurements

# retrieve data 
number_shots = 10000 
L = 10
depth = L # samples will have depth = L-1 since they exclude very last layer containing final measurements
number_circuit_realis = 10
num_meas_rates = 11
num_RNN_units = 64
seed_number = 1
p_val = 2
test_acc_list = []
if p_val == 0:
    p = 0.01 
else:
    p = p_val/(num_meas_rates-1)
np.random.seed(seed_number)
test_acc_list_fixed_p = []
for circuit_iter in range(1,number_circuit_realis+1):
    try:
        measurement_record_0 = dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, L//2)     
        measurement_record_1 = dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, L//2+1) 
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
        test_percentage = 0.2 
        train_percentage = 1 - test_percentage 
        number_samples = len(measurement_records)
        train_data_number_samples = round(train_percentage * number_samples)
        train_data = data[0:train_data_number_samples,:,:]
        train_labels = labels[0:train_data_number_samples]
        test_data = data[train_data_number_samples:number_samples,:,:]
        test_labels = labels[train_data_number_samples:number_samples]
        # Define the RNN model
        model = tf.keras.Sequential() 
        model.add(tf.keras.layers.Input(shape = (depth-1, L)))
        model.add(tf.keras.layers.LSTM(units = num_RNN_units))
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

        # Compile the model with binary crossentropy loss and Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Train the model
        model.fit(train_data, train_labels, epochs=10, batch_size=32) 

        # test the model
        test_loss, test_acc = model.evaluate(test_data, test_labels)
        print('circuit_reali', circuit_iter, 'Test accuracy:', test_acc)
        #model.summary()
        test_acc_list_fixed_p.append(test_acc)
        #test_acc_arr[circuit_iter-1,p_val] = test_acc
    except:
        print(" ignore circuit iter ", circuit_iter)  
print(" number of circuit realis ", len(test_acc_list_fixed_p))  
print(test_acc_list_fixed_p)
np.save("learnability_transitions_cluster/data/accuracy_LSTM_decoder_{}_L_{}_p_{}_number_shots_{}.npy".format(num_RNN_units,L,p,number_shots), test_acc_list_fixed_p) # store all measurements except for final measurements 
