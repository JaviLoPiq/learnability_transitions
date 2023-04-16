import tensorflow as tf
from tf.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #do not show tensorflow warnings
import numpy as np 

# retrieve data 
number_shots = 20000 
L = 18
depth = L-1 # samples will have depth = L-1 since they exclude very last layer containing final measurements
number_circuit_realis = 10
num_meas_rates = 11
num_RNN_units = 64
test_acc_list = []
for p_val in range(0,num_meas_rates):
    p = p_val/(num_meas_rates-1)
    print('meas_rate', p)
    test_acc_list_fixed_p = []
    for circuit_iter in range(1,number_circuit_realis+1):
        try:
            measurement_record_0 = np.load("learnability_transitions_cluster/data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,0,number_shots,circuit_iter))
            measurement_record_1 = np.load("learnability_transitions_cluster/data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,1,number_shots,circuit_iter))
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
            model.add(tf.keras.layers.Input(shape = (depth, L)))
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
    print(" number of circuit realis ", len(test_acc_list))  
    test_acc_list.append(test_acc_list_fixed_p)   

np.save("plots/test_accuracy_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(num_RNN_units,L,number_shots), test_acc_list) # store all measurements except for final measurements 
