import numpy as np
import qiskit as qk
import random
import pickle
import h5py
from itertools import combinations

from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector

## Global parameters
PARAMS_PER_GATE = 6 # number parameters for general U1 2q gate

def u1gate(circ, gate_params, q1, q2, debug=False):
    """
    Function to create U(1) 2-qubit gate; see https://arxiv.org/pdf/quant-ph/0308006.pdf

    Args:
        circ (QuantumCircuit): Qiskit circuit containing qubits q1, q2.
        gate_params (np.array): Array of PARAMS_PER_GATE=6 floats.
        q1, q2 (QuantumRegister): Qiskit qubits.

    Returns:
        Nothing, adds u1 gate directly to circ.
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

def unitary_gate_from_params(gate_params):
    """
    Generate U(1) unitary matrix for a given set of parameters.
    """
    circ = qk.QuantumCircuit(2, name="U1_gate")
    u1gate(circ,gate_params,1,0) # reverse order of qubits (to go from little-endian convention used by default in qiskit to big-endian convention - i.e. q2q1 to q1q2)
    backend = qk.Aer.get_backend('unitary_simulator')
    job = qk.execute(circ, backend)
    decimals = 10
    return np.array(job.result().get_unitary(circ,decimals))

def dict_to_array_measurements(L, depth, p, circuit_iter, number_shots, Q):
    m_locs = np.load('learnability_transitions_cluster/data/measurement_locs_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), allow_pickle=True)
    # pickle for dict data type
    with open('learnability_transitions_cluster/data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy'.format(L,p,Q,number_shots,circuit_iter), 'rb') as f:
        measurement_outcomes_dict = pickle.load(f)
    measurement_outcomes = [key for key, value in measurement_outcomes_dict.items() for i in range(value)] # list of measurement outcomes, including repeated
    #measurement_record = np.zeros((len(measurement_outcomes_dict),depth,L))
    measurement_record = np.zeros((number_shots, depth, L))
    len_measurements = depth*L+(depth-1) # length of each measurement record when stored as keys 
    ind_measurement = 0
    for measurement in measurement_outcomes: # loop over repeated samples
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
    return measurement_record[:,0:depth-1,:] # return all but last layer, which contains measurements of all qubits

def fixed_charge_state(L,Q): #TODO: remove initial_state(L,Q) from sep and quantum decoders
    """
    Initial superposition of states of fixed charge (Hamming weight)
    """
    state = np.zeros((2,)*L)
    for positions in combinations(range(L), Q):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state[tuple(p)] = 1 
    state = state/np.abs(np.sum(state))**0.5
    return state

class U1MRC(object):
    def __init__(self, number_qubits, depth, measurement_locs, params, initial_charge, debug = False):
        """
        Args:
            number_qubits (int): Number of qubits/system size.
            depth (int): Number of circuit layers (one layer = even or odd bond gates, not both).
            m_locs (np.array): m_locs[i,j]=1 => add measurement after layer i and on qubit j, otherwise 0.
            init_state (np.array): Array with all qubit locations that have state 1 (the rest being in state 0).
            params (np.array): Nested list of circuit parameters,
                params[i][j] is an np.array of PARAMS_PER_GATE=6 floats
                specifying the circuit parameters for the jth gate in layer i (counting from the left of the circuit).
        """

        self.number_qubits = number_qubits
        self.depth = depth
        self.measurement_locs = measurement_locs
        self.params = params
        self.initial_charge = initial_charge
        self.debug = debug

    def state_preparation(self, scrambled_state=False, save_unitaries=False):
        """
        State preparation: returns evolved state by unitaries, when initial state is product state of fixed charge.

        Args: 
            save_unitaries (bool): If True, store unitaries used to prepare state.

        Returns: 
            qreg (QuantumRegister): Quantum registers. 
            circ (QuantumCircuit): Circuit used. 
            U_gates_list (list): List of U(1) unitaries used. It's a list of lists, with odd (even) entries corresponding to odd (even) layers.
        """
        qreg = qk.QuantumRegister(self.number_qubits,'q') #not strictly needed
        circ = qk.QuantumCircuit(qreg) 
        if scrambled_state:
            U_gates_list = []
            charge_locations = random.sample(range(self.number_qubits), self.initial_charge) # randomly select locations of charges
            for i in range(self.initial_charge):
                circ.x([charge_locations[i]])
            # create the circuit layer-by-layer
            for i in range(self.depth):
                U_gates_list_fixed_layer = []
                # gates
                if i%2 == 0: # even layer
                    for j in range(self.number_qubits//2):
                        u1gate(circ, self.params[i][j], qreg[2*j], qreg[2*j+1]) # TODO: same as passing 2j, 2j+1?
                        if save_unitaries: 
                            U_gates_list_fixed_layer.append((2*j, 2*j+1, unitary_gate_from_params(self.params[i][j])))
                else: # odd layer
                    for j in range(1,self.number_qubits//2):
                        u1gate(circ, self.params[i][j-1], qreg[2*j-1], qreg[2*j])
                        if save_unitaries: 
                            U_gates_list_fixed_layer.append((2*j-1, 2*j, unitary_gate_from_params(self.params[i][j-1])))
                if save_unitaries:
                    U_gates_list.append(U_gates_list_fixed_layer)  
            if save_unitaries:
                return qreg, circ, U_gates_list 
            else: 
                return qreg, circ
        else:
            state = fixed_charge_state(self.number_qubits, self.initial_charge)
            sv = Statevector(state.reshape(2**self.number_qubits, 1))
            circ.initialize(sv)
            return qreg, circ


    # TODO: remove measurement_rate as it's only used for storing state. Can we write everything w/o using quantum registers? 
    def generate_u1mrc(self, measurement_rate, reali=1, save_initial_state=False, save_unitaries=False):
        """
        Function to generate a random U(1) circuit.

        Args:
            measurement_rate (float): measurement rate.
            reali (int): circuit realization.
            save_state (bool): if true, save initial state (evolved by unitaries alone) in file.

        Returns:
            circ (qk.QuantumCircuit): quantum circuit including measurements applied to initial state.
        """
        if save_initial_state: # state is scrambled
            if save_unitaries:
                qreg, circ, U_gates_list = self.state_preparation(scrambled_state=True, save_unitaries=True)
            else:
                qreg, circ = self.state_preparation(scrambled_state=True)

            # Obtain the statevector
            backend = Aer.get_backend('statevector_simulator')
            result = execute(circ, backend).result()
            statevector = result.get_statevector(circ)
            with h5py.File('learnability_transitions_cluster/data/initial_state_L_{}_p_{}_Q_{}_iter_{}.hdf5'.format(self.number_qubits, measurement_rate, self.initial_charge, reali), 'w') as f:
                f.create_dataset('statevector', data=np.array(statevector))   
        else:
            U_gates_list = []
            qreg, circ = self.state_preparation()

        creg_list = [qk.ClassicalRegister(self.number_qubits,'c'+str(j)) for j in range(self.depth)] # for measurement outcomes

        for reg in creg_list:
            circ.add_register(reg)
        if len(self.params) == self.depth: # no scrambling layers!
            initial_layer = 0 
            final_layer = self.depth 
        else: # the circuit contains scrambling layers 
            initial_layer = self.depth 
            final_layer = 2*self.depth
        # apply layers of unitaries + measurements    
        for i in range(initial_layer, final_layer): 
            U_gates_list_fixed_layer = []
            # gates
            # TODO: avoid using save_unitaries twice!
            if i%2 == 0: # even layer
                for j in range(self.number_qubits//2):
                    u1gate(circ, self.params[i][j], qreg[2*j], qreg[2*j+1], debug=False)
                    if save_unitaries: 
                        U_gates_list_fixed_layer.append((2*j, 2*j+1, unitary_gate_from_params(self.params[i][j])))
            else: # odd layer
                for j in range(1,self.number_qubits//2):
                    u1gate(circ, self.params[i][j-1], qreg[2*j-1], qreg[2*j], debug=False)
                    if save_unitaries: 
                        U_gates_list_fixed_layer.append((2*j-1, 2*j, unitary_gate_from_params(self.params[i][j-1])))
            if initial_layer + i < final_layer - 1:
                for j in range(self.number_qubits):
                    if self.measurement_locs[i-initial_layer,j]: 
                        circ.measure(j, creg_list[i-initial_layer][j])
            if save_unitaries:            
                U_gates_list.append(U_gates_list_fixed_layer)            
            if self.debug: circ.barrier()

        # final measurements
        circ.measure(qreg,creg_list[i-initial_layer])
        # TODO: think of better way of dealing with save_unitaries
        if save_unitaries:
            return circ, U_gates_list
        else:
            return circ
