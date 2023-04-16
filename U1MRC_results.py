import numpy as np
import qiskit as qk
from U1MRC import U1MRC

PARAMS_PER_GATE = 6
L = 6 # number of qubits
depth = L 
number_shots = 500 # number of samples collected
save_state = True # save initial state if True

for p_val in range(0,1):
    p = p_val/10.0
    for circuit_iter in range(1,2):
        # Random parts
        # measurement locations
        # entry [i,j] is layer i, qubit j
        # 1 denotes measurement, 0 = no measurement
        m_locs = np.random.binomial(1,p,L*(depth-1)).reshape((depth-1,L))

        # generate random circuit parameters
        # each layer has L//2
        #params = 4*np.pi*np.random.rand(depth,L//2,PARAMS_PER_GATE)
        param_list = [[4*np.pi*np.random.rand(PARAMS_PER_GATE) 
                    for j in range(L//2-(i%2))] # there are either L//2 or L//2-1 gates for even/odd layers resp.
                    for i in range(depth)]

        for s in range(0,2):
            # initial charge
            initial_charge = int(L/2)+s

            # Create a circuit instance
            # Set debug to False to avoid printing out circuit
            u1mrc = U1MRC(number_qubits=L, depth=depth, measurement_locs=m_locs, params=param_list, initial_charge=initial_charge, debug=False)
            circ = u1mrc.generate_u1mrc(measurement_rate=p, reali=circuit_iter, save_state=save_state)
            backend = qk.Aer.get_backend('qasm_simulator')
            job = qk.execute(circ, backend, shots=number_shots)
            # retrieve measurement outcomes as configurations of \pm 1 if measurement outcome 1/0, and 0 if no measurement applied
            # to do so we need the measurement outcomes from get_counts + measurement locations (to distinguish 0 from no measurement)
            measurement_outcomes = job.result().get_counts(circ)
            """
            print(measurement_outcomes)
            for measurement in measurement_outcomes: 
                print(measurement)
            number_different_outcomes = len(measurement_outcomes)
            measurement_record = np.zeros((number_different_outcomes,depth,L))
            frequency_measurements = np.zeros(number_different_outcomes)
            ind_proba = 0 
            for frequency in measurement_outcomes.values(): 
                frequency_measurements[ind_proba] = frequency
                ind_proba += 1 
            len_measurements = depth*L+(depth-1) # length of each measurement record when stored as keys 
            ind_measurement = 0
            for measurement in measurement_outcomes: # measurement record as keys
                ind_qubit = 0 
                ind_layer = 0
                for i in range(len_measurements): 
                    if ind_qubit != L: 
                        if (i < (L+1)*(depth-1)): # read bitstrings backwards (left most ones correspond to last layer)
                            if m_locs[ind_layer,ind_qubit]:
                                measurement_record[ind_measurement,ind_layer,ind_qubit] = int(2*(int(measurement[len_measurements-i-1])-1/2)) # measurement outcome \pm 1 
                            else:
                                measurement_record[ind_measurement,ind_layer,ind_qubit] = 0 # measurement records only on first depth-1 layers
                        else: 
                            measurement_record[ind_measurement,ind_layer,ind_qubit] = int(2*(int(measurement[len_measurements-i-1])-1/2)) # measurement outcome \pm 1 
                        ind_qubit += 1
                    else: 
                        ind_qubit -= L
                        ind_layer += 1
                ind_measurement += 1    
            print(p, circuit_iter, s)    
            #circ.draw(output='mpl',scale=0.3)
            np.save("learnability_transitions_cluster/data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,s,number_shots,circuit_iter), measurement_record[:,0:depth-1,:]) # store all measurements except for final measurements 
            """
            np.save("learnability_transitions_cluster/data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,s,number_shots,circuit_iter), measurement_outcomes) # store all measurements except for final measurements 