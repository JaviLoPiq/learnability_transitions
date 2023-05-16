import numpy as np
import qiskit as qk
from U1MRC import U1MRC
import h5py
import pickle

PARAMS_PER_GATE = 6
L = 10 # number of qubits
depth = L 
number_shots = 1000 # number of samples collected
save_initial_state = False # save initial state if True
save_unitaries = True # save unitaries used throughout entire circuit (including scrambling step if used) if True (set True by default)
circuit_iter = 3
p = 0.2

np.random.seed(circuit_iter)
# Random parts
# measurement locations
# entry [i,j] is layer i, qubit j
# 1 denotes measurement, 0 = no measurement
m_locs = np.random.binomial(1,p,L*(depth-1)).reshape((depth-1,L))

np.save('learnability_transitions_cluster/data/measurement_locs_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), m_locs)
if save_initial_state: # i.e. if we initialize state via scrambling unitaries
    total_depth = 2*depth 
else:
    total_depth = depth 
# generate random circuit parameters    
param_unitaries_list = [[4*np.pi*np.random.rand(PARAMS_PER_GATE) 
            for j in range(L//2-(i%2))] # there are either L//2 or L//2-1 gates for even/odd layers resp.
            for i in range(total_depth)] # depth (initialization/scrambling) + depth (measurements)

for s in range(0,2):
    # initial charge 
    Q = L//2 + s
    #quantum_dynamics_2(Q,L,p,depth_ratio=depth_ratio,scrambling_type=scrambling_type,is_noisy=is_noisy,decoding_protocol=decoding_protocol)
    u1mrc = U1MRC(number_qubits=L, depth=depth, measurement_locs=m_locs, params=param_unitaries_list, initial_charge=Q, debug=False)
    if save_unitaries:
        circ, U_list = u1mrc.generate_u1mrc(measurement_rate=p, reali=circuit_iter, save_initial_state=save_initial_state, save_unitaries=True)
        if s == 0: # need only save unitaries for one of the charges
            np.save('learnability_transitions_cluster/data/unitaries_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), np.array(U_list, dtype=object))
    else:
        circ = u1mrc.generate_u1mrc(measurement_rate=p, reali=circuit_iter, save_initial_state=save_initial_state, save_unitaries=False)
    backend = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(circ, backend, shots=number_shots)
    # retrieve measurement outcomes as configurations of \pm 1 if measurement outcome 1/0, and 0 if no measurement applied
    # to do so we need the measurement outcomes from get_counts + measurement locations (to distinguish 0 from no measurement)
    measurement_outcomes = job.result().get_counts(circ)
    # Save the dictionary to a file
    with open('learnability_transitions_cluster/data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy'.format(L,p,Q,number_shots,circuit_iter), 'wb') as f:
        pickle.dump(dict(measurement_outcomes), f)      
    #np.save("data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,s,number_shots,circuit_iter), dict(measurement_outcomes)) # store all measurements except for final measurements 
