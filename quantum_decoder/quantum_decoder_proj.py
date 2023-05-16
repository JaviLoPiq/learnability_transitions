import time
import numpy as np
import pickle
from itertools import combinations
import qiskit as qk
import sys
import h5py 
sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC, fixed_charge_state
circuit_reali = 1

def initial_state(L):
    """
    Initial superposition of states of fixed charge (Hamming weight)
    """
    state1 = np.zeros((2,)*L)
    for positions in combinations(range(L), L//2):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state1[tuple(p)] = 1 
    state1 = state1/np.sum(np.abs(state1)**2)**0.5   
    state2 = np.zeros((2,)*L)
    for positions in combinations(range(L), L//2+1):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state2[tuple(p)] = 1   
    state2 = state2/np.sum(np.abs(state2)**2)**0.5            
    state = (state1+state2)/2 # P(Q) = P(Q2)
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


def quantum_dynamics_2(data, Q, U_list, initial_state_0, initial_state_1, decoding_protocol=0):
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

    (depth,L) = data.shape # depth (of unitaries + measurements) is L-1 usually (last layer does not contain measurements)
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
    indices_Q2 = get_indices(L,Q2)
    p_success = []

    state_Q = initial_state_0.copy()
    state_Q2 = initial_state_1.copy()

    state_Q = initial_state(L) # TODO: Fix this so that no need to do equal superposition!
    traj = data
    state_Q_is_zero = False
    state_Q2_is_zero = False

    T = len(U_list) # includes t_scr and depth
    log_Z = []
    log_Z2 = []
    if T == depth + 1: # if no scrambling step 
        initial_layer = 0 
    else: 
        initial_layer = depth + 1 
    for t in range(T-1):
        for x,y,U in U_list[t]:
            if t >= initial_layer: 
                outcomes = (traj[t-initial_layer,x],traj[t-initial_layer,y])
                state_Q, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_Q_is_zero,L,do_measure=True,debug=False)
                #state_Q2, state_Q2_is_zero = unitary_measurement(x,y,U,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,do_measure=True,debug=False)
            #else: #scrambling step
            #    outcomes = None
            #    state, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state,log_Z,state_Q_is_zero,L,do_measure=False,debug=False)                    

            
            p_Q = get_probability(state_Q,L,Q,indices=indices_Q)
            p_Q2 = get_probability(state_Q,L,Q2,indices=indices_Q2)

            if (p_Q == 0) and (p_Q2 == 0): #TODO: revise!
                raise ValueError("can't be both p_Q and p_Q2 zero")
                
        if t >= initial_layer: 
            """
            if state_Q_is_zero:
                if state_Q2_is_zero: #TODO: chech that if both state_Q_is_zero and state_Q2_is_zero => p_Q = 1
                    raise ValueError("can't be both p_Q and p_Q2 zero")
                    
                else:
                    p_Q = 0     
                    p_success.append(p_Q)
            else:   
                ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2)) # ratio of the prob. is the ratio of the respective partition function in the SEP dynamics
                p_Q = 1/(1+1/ratio_p)    
                p_success.append(p_Q)
            """
            p_success.append(p_Q)    
    #print(p_success)                    
    return p_success