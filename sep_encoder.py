import numpy as np 
import random 
from sep_decoder.sep_decoder_proj import initial_state 

def measurement_outcome(state_Q, x):

    # Fix the specified axis to 1
    spin_up = 1
    slices_spin_up = [slice(None)] * L
    slices_spin_up[x] = spin_up
    fixed_array = state_Q[tuple(slices_spin_up)]

    # Get the indices of non-zero elements along the remaining axes
    nonzero_indices = np.nonzero(fixed_array)
    # Access the non-zero elements
    nonzero_elements = fixed_array[nonzero_indices]
    Born_proba_up = np.sum(nonzero_elements)

    # Fix the specified axis to 0
    spin_down = 0
    slices_spin_down = [slice(None)] * L
    slices_spin_down[x] = spin_down
    fixed_array = state_Q[tuple(slices_spin_down)]

    # Get the indices of non-zero elements along the remaining axes
    nonzero_indices = np.nonzero(fixed_array)

    # Access the non-zero elements
    nonzero_elements = fixed_array[nonzero_indices]    
    Born_proba_down = np.sum(nonzero_elements)
    Born_proba = Born_proba_up/(Born_proba_up + Born_proba_down) # Born proba up spin at site x
    
    assert np.abs(Born_proba_down + Born_proba_up - 1.0) < 1E-6

    if random.random() < Born_proba: # spin up
        outcome = 1 
        state_Q[tuple(slices_spin_down)] = 0.0
    else:
        outcome = -1 
        state_Q[tuple(slices_spin_up)] = 0.0

    sQ = np.sum(state_Q)
    state_Q = state_Q/sQ 

    return outcome, state_Q 


def transfer(state_Q, T, x):    
    if x%2 == 1: # transfer step in odd layer
        state_Q = np.moveaxis(state_Q,0,-1)

    state_Q = state_Q.reshape((4,)*(L//2)) #TODO: do we need to reshape back and forth for unitary layer?
    state_Q = np.tensordot(T,state_Q,axes=(-1,x//2))
    state_Q = np.moveaxis(state_Q,0,x//2) # moving the axis back to x//2 as the result from tensordot will be stored in axis=0
    state_Q = state_Q.reshape((2,)*L)

    if x%2 == 1:
        state_Q = np.moveaxis(state_Q,-1,0) 
    
    return state_Q    

L = 8
Q = 4
depth = L//2
x = 2
T = np.eye(4)
state_Q = initial_state(L,Q)
#print(state_Q[:,:,1,:])
#print(measurement_outcome(state_Q,3))


meas_rate = 0.2
num_meas_outcomes = 100

meas_outcomes = np.zeros((num_meas_outcomes, L, depth-1))
# Transfer matrix
T = np.eye(4)
T[1,1] = 1/2
T[2,2] = 1/2
T[1,2] = 1/2
T[2,1] = 1/2

for meas in range(num_meas_outcomes): 
    for t in range(depth-1)[:]:
        
        if t%2 == 0: #even layer
            for x in range(0,L-1,2):
                state_Q = transfer(state_Q, T, x)
        else:
            for x in range(1,L-1,2):
                state_Q = transfer(state_Q, T, x)

        for x in range(L):
            if random.random() < meas_rate:
                meas_outcomes[meas,x,t], state_Q = measurement_outcome(state_Q, x)

#print(meas_outcomes)

