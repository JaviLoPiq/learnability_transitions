import tensorflow as tf
from tensorflow.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #do not show tensorflow warnings
import numpy as np 
import sys
from keras.callbacks import History
from keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC, dict_to_array_measurements

# retrieve data 
number_shots = 2500
percentage_samples_used = 1
L = 10
depth = L # samples will have depth = L-1 since they exclude very last layer containing final measurements
number_circuit_realis = 10
num_meas_rates = 11
num_RNN_units = 64
num_RNN_units_2 = 64
learning_rate = 1E-3
num_epochs = 3
batch_size = 32
dropout=0.3
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
        number_samples = len(measurement_records) * percentage_samples_used
        train_data_number_samples = round(train_percentage * number_samples)
        train_data = data[0:train_data_number_samples,:,:]
        train_labels = labels[0:train_data_number_samples]
        test_data = data[train_data_number_samples:number_samples,:,:]
        test_labels = labels[train_data_number_samples:number_samples]
        # Define the RNN model
        model = tf.keras.Sequential() 
        model.add(tf.keras.layers.Input(shape = (depth-1, L)))
        # add layers: all but last one should have "return_sequences=True" as argument
        model.add(tf.keras.layers.LSTM(units = num_RNN_units, dropout=dropout))
        #model.add(tf.keras.layers.LSTM(units = num_RNN_units_2))
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

        optimizer = Adam(learning_rate=learning_rate)

        # Compile the model with binary crossentropy loss and Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        
        # Train the model
        history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_data, test_labels), callbacks=[early_stopping])

        val_accuracy = history.history['val_accuracy']

        stopped_epoch = early_stopping.stopped_epoch

        training_loss = history.history['loss']
        testing_loss = history.history['val_loss']
        test_acc = history.history['val_accuracy']

        stopped_epoch = early_stopping.stopped_epoch

        test_acc_list_fixed_p.append(val_accuracy)
        # alternatively, use:
        #predictions = model.predict(test_data)
        #test_acc = np.mean([abs(predictions[i][0] - test_labels[i]) < 0.5 for i in range(len(test_labels))])
        print('circuit_reali', circuit_iter, 'Test accuracy:', test_acc)
        #model.summary()
        test_acc_list_fixed_p.append(test_acc)
    except:
        print(" ignore circuit iter ", circuit_iter)  
    print(test_acc_list_fixed_p)#, np.mean(test_acc_list_fixed_p))
    #np.save("learnability_transitions_cluster/data/accuracy_LSTM_decoder_{}_layer_2_32_L_{}_p_{}_number_shots_{}.npy".format(num_RNN_units,L,p,number_shots), test_acc) # store all measurements except for final measurements 
