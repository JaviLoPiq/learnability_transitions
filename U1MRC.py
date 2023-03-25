import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt

## Global parameters
PARAMS_PER_GATE = 6 # number parameters for general U1 2q gate

import random
# Function to create 2q gate
def u1gate(circ,gate_params,q1,q2,debug=False):
    """
    inputs: 
        circ = qiskit circuit containing q1,q2
        gate_parmas = np.array of PARAMS_PER_GATE=6 floats
        q1,q2 qiskit qubits
    returns:
        nothing, adds u1 gate directly to circ
    """
    if debug: # for debugging circuits, just put in czs for ease of visualizing
        circ.cz(q1,q2) 
    else:
        # arbitrary z rotations
        circ.rz(gate_params[0],q1)
        circ.rz(gate_params[1],q2)

        # XX+YY,ZZ rotations
        circ.rz(np.pi/2,q2)
        circ.cnot(q2,q1)
        circ.rz(2*gate_params[2]-np.pi/2,q1)
        circ.ry(np.pi/2-2*gate_params[3],q2)
        circ.cnot(q1,q2)
        circ.ry(2*gate_params[3]-np.pi/2,q2)
        circ.cnot(q2,q1)
        circ.rz(-np.pi/2,q1)

        # arbitrary z rotations    
        circ.rz(gate_params[4],q1)
        circ.rz(gate_params[5],q2)

# state preparation: returns evolved state by unitaries, when initial state is product state of fixed charge
def state_preparation(L,depth,params,initial_charge):
    charge_locations = random.sample(range(L), initial_charge) # randomly select locations of charges
    qreg = qk.QuantumRegister(L,'q')
    # add the registers to the circuit
    circ = qk.QuantumCircuit(qreg)
    for i in range(initial_charge):
        circ.x([charge_locations[i]])
    # create the circuit layer-by-layer
    for i in range(depth):
        # gates
        if i%2 == 0: # even layer
            for j in range(L//2):
                u1gate(circ,params[i][j],qreg[2*j],qreg[2*j+1])
        else: # odd layer
            for j in range(1,L//2):
                u1gate(circ,params[i][j-1],qreg[2*j-1],qreg[2*j])    
    return qreg, circ                      

# Function to generate a random circuit
def generate_u1mrc(L,depth,m_locs,params,initial_charge,debug=False):
    """
    inputs:
        - L, int, system size
        - depth, int, number of circuit layers (one layer = even or odd bond gates, not both)
        - m_locs, np.array of bools, m_locs[i,j]=1 => add measurement after layer i and on qubit j
        - init_state, np.array with all qubit locations that have state 1 (the rest being in state 0)
        - params, nested list of circuit parameters, 
            params[i][j] is an np.array of PARAMS_PER_GATE=6 floats
            specifying the circuit parameters for the jth gate in layer i (counting from the left of the circuit)
        - debug, bool, if True replaces u1 gates with cz gates and adds barriers so that you can visualize more easily
    outputs:
        - qiskit circuit of appropriate
    """
    qreg, circ = state_preparation(L,depth,params,initial_charge)
    creg_list = [qk.ClassicalRegister(L,'c'+str(j)) for j in range(depth)] # for measurement outcomes

    for reg in creg_list:
        circ.add_register(reg)
    # create the circuit layer-by-layer
    for i in range(depth):
        # gates
        if i%2 == 0: # even layer
            for j in range(L//2):
                u1gate(circ,params[i][j],qreg[2*j],qreg[2*j+1],debug=debug)
        else: # odd layer
            for j in range(1,L//2):
                u1gate(circ,params[i][j-1],qreg[2*j-1],qreg[2*j],debug=debug)
        if i<depth-1:        
            for j in range(L):
                if m_locs[i,j]:
                    circ.measure(j,creg_list[i][j])       

        if debug: circ.barrier()

    # final measurements
    circ.measure(qreg,creg_list[i])
    
    return qreg, creg_list, circ

L = 12
depth = L
for p_val in range(0,11):
    p = p_val/10.0
    for circuit_iter in range(1,11):
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
            # Draw circuit
            print('Remember to set debug=FALSE to generate actual circuits...')
            qreg,creg_list,circ = generate_u1mrc(L,depth,m_locs,param_list,initial_charge,debug=False)
            backend = qk.Aer.get_backend('qasm_simulator')
            number_shots = 500
            job = qk.execute(circ, backend, shots=number_shots)
            # retrieve measurement outcomes as configurations of \pm 1 if measurement outcome 1/0, and 0 if no measurement applied
            # to do so we need the measurement outcomes from get_counts + measurement locations (to distinguish 0 from no measurement)
            measurement_outcomes = job.result().get_counts(circ)
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
            np.save("measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,s,number_shots,circuit_iter), measurement_record[:,0:depth-1,:]) # store all measurements except for final measurements 