import tensorflow as tf
from tensorflow.keras import layers
import numpy as np 

# Define the CNN model
def create_model(input_shape):
    model = tf.keras.Sequential()
    # Add a convolutional layer with 32 filters and 3x3 kernel size
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    # Add a max pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    # Add another convolutional layer with 64 filters and 3x3 kernel size
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    # Add a max pooling layer
    model.add(layers.MaxPooling2D((2,2)))
    # Flatten the output of the convolutional layers
    model.add(layers.Flatten())
    # Add a dense layer with 64 units
    model.add(layers.Dense(64, activation='relu'))
    # Add the output layer with a single unit and sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# retrieve data 
number_shots = 1000 
L = 8
depth = L-1 # samples will have depth = L-1 since they exclude very last layer containing final measurements
test_acc_arr = np.zeros((10,11)) # number of circuit iterations, number of p's
num_circuit_realis = 10
num_meas_rates = 11
for p_val in range(0,num_meas_rates):
    p = p_val/(num_meas_rates-1)
    print('meas_rate', p)
    for circuit_iter in range(1,num_circuit_realis+1):
        measurement_record_0 = np.load("../data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,0,number_shots,circuit_iter))
        measurement_record_1 = np.load("../data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,1,number_shots,circuit_iter))
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


        # Create the model with input shape (n, 1)
        model = create_model(input_shape=(depth,L,1))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #measurement_records = np.zeros(2,np.shape(measurement_record_0))

        # Train the model
        model.fit(train_data, train_labels, epochs=10, batch_size=32) 

        # test the model
        test_loss, test_acc = model.evaluate(test_data, test_labels)
        print('circuit_reali', circuit_iter, 'Test accuracy:', test_acc)
        test_acc_arr[circuit_iter-1,p_val] = test_acc 

np.save("../data/test_accuracy_CNN_L_{}_numbershots_{}.npy".format(L,number_shots), test_acc_arr) # store all measurements except for final measurements 