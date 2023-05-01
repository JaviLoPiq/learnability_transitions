import time
import numpy as np
import pickle
from itertools import combinations
import qiskit as qk
import sys
import h5py 
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC
circuit_reali = 1


def initial_state(L, Q):

    """ remove this
    state = np.zeros((2,)*L)
    Q_state = [1,0]*(L//2)
    """
    # assuming Q2 = L//2 -1
    #Q2_state = [1,0]*(L//2 - 1) + [0,0]
    """
    # assuming Q2 = L//2 + 1
    Q2_state = [1,0]*(L//2 - 1) + [1,1]
    state[tuple(Q_state)] = 1/2**0.5
    #state[tuple(Q2_state)] = 1/2**0.5
    """

    state = np.zeros((2,)*L)
    for positions in combinations(range(L), Q):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state[tuple(p)] = 1 
    state = state/np.abs(np.sum(state))**0.5

    return state

def unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_zero,L,do_measure,debug=False):
    """_summary_

    Args:
        x,y(_type_): transfer matrix is applied on qubit x, y

        U (_type_): Unitary matrix.
        outcomes (_type_): measurement outcomes at x, x+1. outcome has 3 possible values, 0: no measurement, 1: measured outcome is 1, -1: measured outcome is 0
        state_Q (_type_): state of the system
        log_Z (_type_): list of log of the partition functions in the previous steps
        state_zero (Boolean): if true then state_Q has zero partition function/weight; the measurement outcomes are compatible with the total charge.
        L (_type_): system size
        do_measure: Boolean variable. If true perform measurement
        debug (bool, optional): Defaults to False.

    Returns:
        state_Q: transferred state
        state_zero: if true, the partition function becomes zero
    """

    if not state_zero:

        state_Q = np.swapaxes(state_Q,y,(x+1)%L) # moving y-axis to the right of the x-axis

        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,0,-1)
        state_Q = state_Q.reshape((4,)*(L//2))
        state_Q = np.tensordot(U,state_Q,axes=(-1,x//2))

        if not do_measure:
            state_Q = np.moveaxis(state_Q,0,x//2)
            state_Q = state_Q.reshape((2,)*L)
            if x%2 == 1:
                state_Q = np.moveaxis(state_Q,-1,0)
            state_Q = np.swapaxes(state_Q,y,(x+1)%L)
            return state_Q, state_zero

        if outcomes[0] !=0: # outcome at x is non-zero
            proj1 = int(1 - (outcomes[0]+1)//2)
            state_Q[2*proj1,:] = 0
            state_Q[2*proj1+1,:] = 0
        if outcomes[1] !=0: # outcome at x+1 in non-zero
            proj2 = int(1 - (outcomes[1]+1)//2)
            state_Q[proj2,:] = 0
            state_Q[proj2+2,:] = 0

        state_Q = np.moveaxis(state_Q,0,x//2)
        state_Q = state_Q.reshape((2,)*L)
        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,-1,0)
        state_Q = np.swapaxes(state_Q,y,(x+1)%L)


        sQ = np.sum(np.abs(state_Q)**2)
        if sQ == 0:
            state_zero = True
            log_Z.append(-np.inf)
        else:
            state_zero = False
            log_Z.append(np.log(sQ))
            state_Q = state_Q/sQ**0.5
    else:
        log_Z.append(-np.inf)

    return state_Q, state_zero


def get_indices(L,Q):
    """
    Get indices of the configuration with total charge equal to Q
    """
    p_list = []
    for positions in combinations(range(L), Q):
            p = [0] * L
            for i in positions:
                p[i] = 1
            p_list.append(np.array(p))

    indices = tuple(np.array(p_list).T)
    return indices

def get_probability(state,L,Q,indices=[]):
    """
    This function return probability of having charge Q in the state
    Args:
        state ((2,)*L shape array): state of the quantum system
        L (_type_): system size
        Q (_type_): Charge

    Returns:
        prob: probability of having charge Q in the state
    """
    if not indices:
        indices = get_indices(L,Q)
    prob = np.sum(np.abs(state[indices])**2)
    return prob


def quantum_dynamics_2(data, Q, U_list, initial_state, decoding_protocol=0):
    """
    input:
        - data (_type_): 2d array of shape (depth-1,L) holding values of the outcomes
        - Q (_type_): initial charge of the quantum state which was used to generate the outcomes
        - neel_initial_state (bool, optional): Defaults to True. This argument is redundant now!
        - decoding_protocol
            0: No active decoding. Post-select such that both P_Q and P_Q1 are not 0
            1: Postselect on trajectories where P_suc != 0, i.e P_Q != 0
            2: Postselect on trajectories where last layer has total charge = Q
            3: Union of 2 and 1

    Returns:
        p_Q: Probility of the initial charge being Q (the true charge) in the SEP dynamics
    """

    (depth,L) = data.shape # depth is L-1 usually (last layer does not contain measurements)
    """
    # assuming charges L//2 and L//2-1
    Q2 = Q-1
    if Q<L//2:
        Q2 = Q+1
    """
    # assuming charge L//2 or L//2 + 1 
    Q2 = Q + 1
    if Q > L//2:
        Q2 = Q - 1
    indices_Q = get_indices(L,Q)

    if decoding_protocol == 2 or decoding_protocol == 3:
        if np.sum((data[-1,:]+1)) != 2*Q:
            print('Alert!',L,Q)
            print(data[:,:],Q)
            return False

    total = 0
    p_success = []

    state = initial_state

    N=1
    traj = data
    state_Q_is_zero = False

    # state_Q2_is_zero = False
    # state_Q = initial_state_Q.copy()
    # state_Q2 = initial_state_Q2.copy()

    T = len(U_list) # includes t_scr and depth

    log_Z = []
    total += N
    for t in range(T)[:-1]:
            for x,y,U in U_list[t]:
                if t >= T-depth: # TODO: or t >= depth?
                    outcomes = (traj[t-T+depth,x],traj[t-T+depth,y])
                    state, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state,log_Z,state_Q_is_zero,L,do_measure=True,debug=False)

                p_Q = get_probability(state,L,Q,indices=indices_Q)

                # assert np.round(p_Q + p_Q2,8) == 1, (p_Q,p_Q2,np.round(p_Q+p_Q2,8))

                # state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)


                if p_Q == 0:
                    if decoding_protocol == 1 or decoding_protocol == 3:
                        return False

            p_success.append(p_Q)

    return p_success


"""
print(np.shape(initial_state(4,2)))
#print(initial_state(4,3))
inds = get_indices(4,3)
state = initial_state(4,3)
print(state[inds])
print(np.sum(np.abs(state[inds])**2))
"""

# run()
L = 10
depth = L
PARAMS_PER_GATE = 6
#p = int(sys.argv[1])/10
p = 0.3
number_shots = 100
depth_ratio = 1
scrambling_type = 'Special'
is_noisy = False
decoding_protocol = 3
circuit_iter = 2
s = 0
Q = L//2 + s 

with h5py.File('data/initial_state_L_{}_p_{}_Q_{}_iter_{}.hdf5'.format(L, p, Q, circuit_iter), 'r') as f:
    # Retrieve the dataset and convert it to a numpy array
    my_array = f['statevector'][()]
    state = np.array(my_array.tolist())
    scrambled_state = state.reshape((2,)*L)


#TODO: replace unitaries by its parameters?
U_list = np.load('data/unitaries_L_{}_p_{}_Q_{}_iter_{}.npy'.format(L, p, s, circuit_iter), allow_pickle=True)
m_locs = np.load('data/measurement_locs_L_{}_p_{}_iter_{}.npy'.format(L, p, circuit_iter), allow_pickle=True)
# pickle for dict data type
with open('data/measurement_record_dict_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy'.format(L,p,s,number_shots,circuit_iter), 'rb') as f:
    measurement_outcomes_dict = pickle.load(f)
measurement_outcomes = [key for key, value in measurement_outcomes_dict.items() for i in range(value)] # list of measurement outcomes, including repeated
measurement_record = np.zeros((number_shots,depth,L))
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

accuracy_list = []
for i in range(number_shots):
    accuracy_list.append(quantum_dynamics_2(measurement_record[i,:,:], Q, U_list, scrambled_state, decoding_protocol=decoding_protocol)[2*(depth-1)])

print(np.mean([i > 0.5 for i in accuracy_list]))
print(np.mean(accuracy_list))
#np.save("data/accuracy_quantum_decoder_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,number_shots,circuit_reali), accuracy_list) # store all measurements except for final measurements 
