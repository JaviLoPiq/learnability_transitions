import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC, dict_to_array_measurements
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions    

# retrieve data 
number_shots = 10000
percentage_samples_used = 1
L = 10
depth = L # samples will have depth = L-1 since they exclude very last layer containing final measurements
number_circuit_realis = 7
num_meas_rates = 11
dropout=0.3
seed_number = 1
p_val = 10
num_epochs = 5
len_sequences = (depth-1) * L
batch_size = 64
test_acc_list = []
if p_val == 0:
    p = 0.01 
else:
    p = p_val/(num_meas_rates-1)
np.random.seed(seed_number)
test_acc_list_fixed_p = []
test_loss_list_fixed_p = []
train_loss_list_fixed_p = []
for circuit_iter in range(number_circuit_realis,number_circuit_realis+1):
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
        number_samples = round(len(measurement_records) * percentage_samples_used)
        train_data_number_samples = round(train_percentage * number_samples)
        train_data = data[0:train_data_number_samples,:,:]
        train_data = train_data.reshape(train_data.shape[0], -1) + 2
        train_labels = labels[0:train_data_number_samples]
        test_data = data[train_data_number_samples:number_samples,:,:]
        test_data = test_data.reshape(test_data.shape[0], -1) + 2
        test_labels = labels[train_data_number_samples:number_samples]

        embed_dim = 64  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 64 # Hidden layer size in feed forward network inside transformer
        vocab_size = 4 # 3 possible values {-1,0,1} -> {1,2,3} with 0 having acting as padding token

        inputs = layers.Input(shape=(len_sequences,))
        embedding_layer = TokenAndPositionEmbedding(len_sequences, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        learning_rate = 1E-4
        custom_optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=custom_optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_data, test_labels), callbacks=[early_stopping])

        val_accuracy = history.history['val_accuracy']
        training_loss = history.history['loss']
        testing_loss = history.history['val_loss']

        stopped_epoch = early_stopping.stopped_epoch

        test_acc_list_fixed_p.append(val_accuracy)
        train_loss_list_fixed_p.append(training_loss)
        test_loss_list_fixed_p.append(testing_loss)
    except:
        print(" ignore circuit iter ", circuit_iter)  
        #print(" number of circuit realis ", len(test_acc_list_fixed_p))  
print(test_acc_list_fixed_p, np.mean(test_acc_list_fixed_p))        