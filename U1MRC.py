import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import random
import pickle 

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

    def state_preparation(self):
        """
        state preparation: returns evolved state by unitaries, when initial state is product state of fixed charge. 
        """
        charge_locations = random.sample(range(self.number_qubits), self.initial_charge) # randomly select locations of charges
        qreg = qk.QuantumRegister(self.number_qubits,'q')
        # add the registers to the circuit
        circ = qk.QuantumCircuit(qreg)
        for i in range(self.initial_charge):
            circ.x([charge_locations[i]])
        # create the circuit layer-by-layer
        for i in range(self.depth):
            # gates
            if i%2 == 0: # even layer
                for j in range(self.number_qubits//2):
                    u1gate(circ, self.params[i][j], qreg[2*j], qreg[2*j+1])
            else: # odd layer
                for j in range(1,self.number_qubits//2):
                    u1gate(circ, self.params[i][j-1], qreg[2*j-1], qreg[2*j])    
        return qreg, circ      
                    
    # TODO: remove measurement_rate as it's only used for storing state
    def generate_u1mrc(self, measurement_rate, reali=1, save_state=False):
        """
        Function to generate a random U(1) circuit.

        Args:
            measurement_rate (float): measurement rate. 
            reali (int): circuit realization.
            save_state (bool): if true, save initial state (evolved by unitaries alone) in file.

        Returns:
            circ (qk.QuantumCircuit): quantum circuit including measurements applied to initial state.
        """
        qreg, circ = self.state_preparation()
        
        if save_state: 
            # Open a file for writing
            with open('quantum_register_L_{}_p_{}_Q_{}_iter_{}.pkl'.format(self.number_qubits, measurement_rate, self.initial_charge, reali), 'wb') as f:
            # Serialize the quantum register using pickle and write it to the file
                pickle.dump(qreg, f)

        creg_list = [qk.ClassicalRegister(self.number_qubits,'c'+str(j)) for j in range(self.depth)] # for measurement outcomes

        for reg in creg_list:
            circ.add_register(reg)
        # create the circuit layer-by-layer
        for i in range(self.depth):
            # gates
            if i%2 == 0: # even layer
                for j in range(self.number_qubits//2):
                    u1gate(circ, self.params[i][j], qreg[2*j], qreg[2*j+1], debug=False)
            else: # odd layer
                for j in range(1,self.number_qubits//2):
                    u1gate(circ, self.params[i][j-1], qreg[2*j-1], qreg[2*j], debug=False)
            if i < self.depth - 1:        
                for j in range(self.number_qubits):
                    if self.measurement_locs[i,j]:
                        circ.measure(j, creg_list[i][j])       

            if self.debug: circ.barrier()

        # final measurements
        circ.measure(qreg,creg_list[i])
        
        return circ
