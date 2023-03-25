import numpy as np
import matplotlib.pyplot as plt  
L = 8
number_shots = 500
num_circuit_realis = 5
num_meas_rates = 11
#test_acc_arr_CNN = np.load("data/test_accuracy_CNN_L_{}_numbershots_{}.npy".format(L,number_shots)) 
test_acc_arr_RNN_8 = np.load("data/test_accuracy_RNN_L_{}_numbershots_{}_ancilla.npy".format(L,500))
#test_acc_arr_RNN_12 = np.load("test_accuracy_RNN_L_{}_numbershots_{}_ancilla.npy".format(12,1000)) 
#test_acc_CNN_mean = [0]*num_meas_rates
#test_acc_CNN_err = [0]*num_meas_rates 
test_acc_RNN_8_mean = [0]*num_meas_rates
test_acc_RNN_8_err = [0]*num_meas_rates 
test_acc_RNN_12_mean = [0]*num_meas_rates
test_acc_RNN_12_err = [0]*num_meas_rates 
p_arr = [0]*num_meas_rates 
for p in range(0,num_meas_rates): 
    p_arr[p] = p/10.0
    #test_acc_CNN_mean[p] = np.mean(test_acc_arr_CNN[:,p])
    #test_acc_CNN_err[p] = np.sqrt(np.var(test_acc_arr_CNN[:,p])/num_circuit_realis)
    test_acc_RNN_8_mean[p] = np.mean(test_acc_arr_RNN_8[:,p])
    test_acc_RNN_8_err[p] = np.sqrt(np.var(test_acc_arr_RNN_8[:,p])/num_circuit_realis)
    #test_acc_RNN_12_mean[p] = np.mean(test_acc_arr_RNN_12[:,p])
    #test_acc_RNN_12_err[p] = np.sqrt(np.var(test_acc_arr_RNN_12[:,p])/num_circuit_realis)

#plt.errorbar(p_arr, test_acc_CNN_mean, test_acc_CNN_err, label='CNN')
plt.errorbar(p_arr, test_acc_RNN_8_mean, test_acc_RNN_8_err, label='RNN L=12 500 shots')
#plt.errorbar(p_arr, test_acc_RNN_12_mean, test_acc_RNN_12_err, label='RNN L=12 1000 shots')
plt.xlabel('measurement rate p')
plt.ylabel('accuracy')
plt.legend()
plt.show()
