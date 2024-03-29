"""
This code runs the SEP decoder without sparse matrices and instead uses tensor structure
"""

import os
import pickle
import time
import numpy as np
import scipy.sparse as sparse
from itertools import combinations


def initial_state(L,Q):
    """
    Initial superposition of states of fixed charge (Hamming weight)
    """
    state = np.zeros((2,)*L)
    for positions in combinations(range(L), Q):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state[tuple(p)] = 1 
    state = state/np.sum(state)
    return state

def transfer(x,T,outcomes,state_Q,log_Z,state_zero,L,debug=False):
    """
    Apply transfer matrix on qubits x, x+1. 

    Args:
        x (int): Transfer matrix is applied on qubit x, x+1.
        T (np.array): Transfer matrix.
        outcomes (tuple): Measurement outcomes at x, x+1. outcome has 3 possible values, 0: no measurement, 1: measured outcome is 1, -1: measured outcome is 0.
        state_Q (np.array): State of the system.
        log_Z (list): List of log of the partition functions in the previous steps.
        state_zero (bool): If true then state_Q has zero partition function/weight; the measurement outcomes are compatible with the total charge.
        L (int): System size.
        debug (bool, optional): Defaults to False.

    Returns:
        state_Q: Transferred state.
        state_zero: If true, the partition function becomes zero.
    """
    if not state_zero:
        
        if x%2 == 1: # transfer step in odd layer
            state_Q = np.moveaxis(state_Q,0,-1)

        state_Q = state_Q.reshape((4,)*(L//2))
        state_Q = np.tensordot(T,state_Q,axes=(-1,x//2))
        
        if outcomes[0] !=0: # outcome at x is non-zero
            proj1 = int(1 - (outcomes[0]+1)//2)
            state_Q[2*proj1,:] = 0
            state_Q[2*proj1+1,:] = 0
        if outcomes[1] !=0: # outcome at x+1 in non-zero
            proj2 = int(1 - (outcomes[1]+1)//2)
            state_Q[proj2,:] = 0
            state_Q[proj2+2,:] = 0

        state_Q = np.moveaxis(state_Q,0,x//2) # moving the axis back to x//2 as the result from tensordot will be stored in axis=0

        state_Q = state_Q.reshape((2,)*L)
        if x%2 == 1:
            state_Q = np.moveaxis(state_Q,-1,0)

        sQ = np.sum(state_Q)
        if sQ == 0:
            state_zero = True
            log_Z.append(-np.inf)
        else:
            state_zero = False
            log_Z.append(np.log(sQ))
            state_Q = state_Q/sQ
    else:
        log_Z.append(-np.inf)
    
    return state_Q, state_zero

def boundary_measurement(state_Q,outcomes,L,log_Z,state_zero): #TODO: only apply boundary_measurement on 1st and last qubit, not pairs of qubits
    # Left boundary
    state_Q = state_Q.reshape((4,)*(L//2))
    if outcomes[0] != 0: # outcome at x is non-zero
        proj = int(1 - (outcomes[0]+1)//2)
        state_Q[2*proj,:] = 0
        state_Q[2*proj+1,:] = 0
    # Right boundary
    state_Q = np.moveaxis(state_Q,(L-2)//2,0)
    if outcomes[1] != 0: # outcome at x+1 in non-zero
        proj = int(1 - (outcomes[1]+1)//2)
        state_Q[proj,:] = 0
        state_Q[proj+2,:] = 0
    state_Q = np.moveaxis(state_Q,0,(L-2)//2) # moving axis back to (L-2)//2
    state_Q = state_Q.reshape((2,)*L)   

    sQ = np.sum(state_Q)
    if sQ == 0:
        state_zero = True
        log_Z.append(-np.inf)
    else:
        state_zero = False
        log_Z.append(np.log(sQ))
        state_Q = state_Q/sQ
    return state_Q, state_zero

def sep_dynamics_2(data,Q):
    """
    Compute output success probability P(Q_correct|{m}) based on partition function of SEP.

    Args:
        data (np.array): 2d array of shape (depth-1,L) holding values of the outcomes.
        Q (int): Initial charge of the quantum state which was used to generate the outcomes.
        neel_initial_state (bool, optional): Defaults to True.

    Returns:
        p_Q (float): Success probability.
    """
    (depth,L) = data.shape
   
    """
    # assuming charge L//2 or L//2 - 1
    Q2 = Q-1
    if Q<L//2:
        Q2 = Q+1
    """
    # assuming charge L//2 or L//2 + 1 
    Q2 = Q + 1
    if Q > L//2:
        Q2 = Q - 1  
    p_success = []
    
    initial_state_Q = initial_state(L,Q)
    initial_state_Q2 = initial_state(L,Q2)

    traj = data
    state_Q_is_zero = False
    state_Q2_is_zero = False
    state_Q = initial_state_Q.copy()
    state_Q2 = initial_state_Q2.copy()

    # Transfer matrix
    T = np.eye(4)
    T[1,1] = 1/2
    T[2,2] = 1/2
    T[1,2] = 1/2
    T[2,1] = 1/2

    log_Z = []
    log_Z2 = []
    for t in range(depth)[:]:
        
        if t%2 == 0: #even layer
            for x in range(0,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
            
                state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L)
   
        else:
            for x in range(1,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                debug=False

                state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L)

            outcomes = (traj[t,0], traj[t,L-1])
            state_Q, state_Q_is_zero = boundary_measurement(state_Q,outcomes,L,log_Z,state_Q_is_zero)
            state_Q2, state_Q2_is_zero = boundary_measurement(state_Q2,outcomes,L,log_Z2,state_Q2_is_zero)
        
        if state_Q_is_zero:           
            break
    
        if state_Q2_is_zero:
            break
    
    if state_Q_is_zero:
        if state_Q2_is_zero: 
            raise ValueError("can't be both p_Q and p_Q2 zero")  
        else:
            p_Q = 0
    else:
        ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2)) # ratio of the prob. is the ratio of the respective partition function in the SEP dynamics
        p_Q = 1/(1+1/ratio_p)
    return p_Q
