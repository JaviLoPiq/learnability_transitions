import numpy as np
import matplotlib.pyplot as plt  
L = 18
number_shots = 1000
num_circuit_realis = 10
num_meas_rates = 11
#test_acc_arr_mean_RNN_1 = np.load("plots/test_accuracy_mean_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(32,8,1000))
#test_acc_arr_var_RNN_1 = np.load("plots/test_accuracy_var_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(32,8,1000))
test_acc_arr_mean_RNN_2 = np.load("plots/test_accuracy_mean_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,10,1000))
test_acc_arr_var_RNN_2 = np.load("plots/test_accuracy_var_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,10,1000))
test_acc_arr_mean_RNN_3 = np.load("plots/test_accuracy_mean_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,14,1000))
test_acc_arr_var_RNN_3 = np.load("plots/test_accuracy_var_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,14,1000))
test_acc_arr_mean_RNN_4 = np.load("plots/test_accuracy_mean_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,18,1000))
test_acc_arr_var_RNN_4 = np.load("plots/test_accuracy_var_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,18,1000))
#test_acc_arr_mean_RNN_5 = np.load("plots/test_accuracy_mean_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,16,1000))
#test_acc_arr_var_RNN_5 = np.load("plots/test_accuracy_var_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,16,1000))
#test_acc_arr_mean_RNN_6 = np.load("plots/test_accuracy_mean_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,18,1000))
#test_acc_arr_var_RNN_6 = np.load("plots/test_accuracy_var_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,18,1000))
test_acc_arr_RNN_1 = np.load("plots/test_accuracy_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,10,5000), allow_pickle=True)
test_acc_arr_RNN_2 = np.load("plots/test_accuracy_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,14,5000), allow_pickle=True)
test_acc_arr_RNN_3 = np.load("plots/test_accuracy_RNN_LSTM_{}_L_{}_numbershots_{}.npy".format(64,18,5000), allow_pickle=True)

test_acc_arr_var_RNN_2 = [0]*11
test_acc_arr_var_RNN_3 = [0]*11
test_acc_arr_var_RNN_4 = [0]*11
test_acc_arr_var2_RNN_2 = [0]*11
test_acc_arr_var2_RNN_3 = [0]*11
test_acc_arr_var2_RNN_4 = [0]*11
test_acc_arr_mean_RNN_2 = [np.mean(test_acc_arr_RNN_1[i]) for i in range(0,11)]
#test_acc_arr_var_RNN_2 = [np.var(test_acc_arr_RNN_1[i]) for i in range(0,11)]
for i in range(0,11):
    test_acc_arr_var_RNN_2[i] = np.var(test_acc_arr_RNN_1[i])
    test_acc_arr_var_RNN_3[i] = np.var(test_acc_arr_RNN_2[i])
    test_acc_arr_var_RNN_4[i] = np.var(test_acc_arr_RNN_3[i])
    test_acc_arr_var2_RNN_2[i] = np.mean([test_acc_arr_RNN_1[i][j]**4 for j in range(0,len(test_acc_arr_RNN_1[i]))])
    test_acc_arr_var2_RNN_3[i] = np.mean([test_acc_arr_RNN_2[i][j]**4 for j in range(0,len(test_acc_arr_RNN_2[i]))])
    test_acc_arr_var2_RNN_4[i] = np.mean([test_acc_arr_RNN_3[i][j]**4 for j in range(0,len(test_acc_arr_RNN_3[i]))])
test_acc_arr_mean_RNN_3 = [np.mean(test_acc_arr_RNN_2[i]) for i in range(0,11)]
test_acc_arr_mean_RNN_4 = [np.mean(test_acc_arr_RNN_3[i]) for i in range(0,11)]
print(test_acc_arr_RNN_1)
print(test_acc_arr_RNN_2)
print(test_acc_arr_RNN_3)

p_arr = np.arange(0, 1.1, 0.1)
#plt.errorbar(p_arr, test_acc_arr_mean_RNN_1, np.sqrt(test_acc_arr_var_RNN_1/num_meas_rates), label='L=8')
plt.plot(p_arr, [1 - test_acc_arr_var2_RNN_2[i]/(3*test_acc_arr_var_RNN_2[i]**2) for i in range(0,11)])
plt.plot(p_arr, [1 - test_acc_arr_var2_RNN_3[i]/(3*test_acc_arr_var_RNN_3[i]**2) for i in range(0,11)])
plt.plot(p_arr, [1 - test_acc_arr_var2_RNN_4[i]/(3*test_acc_arr_var_RNN_4[i]**2) for i in range(0,11)])
#plt.errorbar(p_arr, test_acc_arr_mean_RNN_2, np.sqrt(test_acc_arr_var_RNN_2), label='L=10')
#plt.errorbar(p_arr, test_acc_arr_mean_RNN_3, np.sqrt(test_acc_arr_var_RNN_3), label='L=14')
#plt.errorbar(p_arr, test_acc_arr_mean_RNN_4, np.sqrt(test_acc_arr_var_RNN_4), label='L=18')
#plt.errorbar(p_arr, test_acc_arr_mean_RNN_5, np.sqrt(test_acc_arr_var_RNN_5/num_meas_rates), label='L=16')
#plt.errorbar(p_arr, test_acc_arr_mean_RNN_6, np.sqrt(test_acc_arr_var_RNN_6/num_meas_rates), label='L=18')
#plt.errorbar(p_arr, test_acc_RNN_12_mean, test_acc_RNN_12_err, label='RNN L=12 1000 shots')
plt.title(" LSTM(64)")
plt.xlabel('measurement rate p')
plt.ylabel('accuracy')
plt.legend()
plt.show()
