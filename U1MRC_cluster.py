import numpy as np
import qiskit as qk
import sys
import h5py
from U1MRC import U1MRC
import pickle

L = 10
depth = L
number_shots = 2000
p = 1.0
circuit_iter = int(sys.argv[1])
PARAMS_PER_GATE = 6

# Random parts
# measurement locations
# entry [i,j] is layer i, qubit j
# 1 denotes measurement, 0 = no measurement
m_locs = np.random.binomial(1,p,L*(depth-1)).reshape((depth-1,L))

np.save('data/measurement_locs_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), m_locs)
# generate random circuit parameters
# each layer has L//2
#params = 4*np.pi*np.random.rand(depth,L//2,PARAMS_PER_GATE)
param_list = [[4*np.pi*np.random.rand(PARAMS_PER_GATE) 
            for j in range(L//2-(i%2))] # there are either L//2 or L//2-1 gates for even/odd layers resp.
            for i in range(depth)]

for s in range(0,2):
    # initial charge 
    Q = L//2 + s
    #quantum_dynamics_2(Q,L,p,depth_ratio=depth_ratio,scrambling_type=scrambling_type,is_noisy=is_noisy,decoding_protocol=decoding_protocol)
    u1mrc = U1MRC(number_qubits=L, depth=depth, measurement_locs=m_locs, params=param_list, initial_charge=Q, debug=False)
    circ, U_list = u1mrc.generate_u1mrc(measurement_rate=p, reali=circuit_iter, save_state=True, save_unitaries=True)
    np.save('data/unitaries_L_{}_p_{}_Q_{}_iter_{}.npy'.format(L, p, s, circuit_iter), np.array(U_list, dtype=object))
    backend = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(circ, backend, shots=number_shots)
    # retrieve measurement outcomes as configurations of \pm 1 if measurement outcome 1/0, and 0 if no measurement applied
    # to do so we need the measurement outcomes from get_counts + measurement locations (to distinguish 0 from no measurement)
    measurement_outcomes = job.result().get_counts(circ)
    # Save the dictionary to a file
    with open('data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy'.format(L,p,s,number_shots,circuit_iter), 'wb') as f:
        pickle.dump(dict(measurement_outcomes), f)      
    #np.save("data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,s,number_shots,circuit_iter), dict(measurement_outcomes)) # store all measurements except for final measurements 
