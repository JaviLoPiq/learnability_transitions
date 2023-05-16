"""
This code runs the SEP decoder without sparse matrices and instead uses tensor structure
"""

import os
import pickle
import time
import numpy as np
import scipy.sparse as sparse
#import matplotlib.pyplot as pl
from itertools import combinations
import sys

L = 10
p = (int(sys.argv[1])-1)/10
number_shots = 10000 

# Function to get the initial state used in the Qiskit simulation
def initial_state(L,Q,neel_state=True):
    

    state = np.zeros((2,)*L)
    for positions in combinations(range(L), Q):
        p = [0] * L

        for i in positions:
            p[i] = 1

        state[tuple(p)] = 1 
    state = state/np.sum(state)
    return state

    # Redundant code below. I am fixing the initial to uniform superposition of all states compatible with the initial charge Q

    if not neel_state:
        filename = 'Weak measurements/data/scrambled_states/L='+str(L)+'_T='+str(2*L)+'_Q='+str(Q)
        with open(filename,'rb') as f:
            state = pickle.load(f)
        state = np.abs(np.asarray(state).reshape((2,)*(L+1))[0,:])**2
        state = np.transpose(state,range(L)[::-1]) # reversing the order of qubits as qiskit has qubit #0 at the right end
    else:
        state = np.zeros((2,)*L)
        confi = [0]*L
        for x in range(0,Q,1):
            confi[2*x] = 1
        state[tuple(confi)] = 1
    
    #
    #
    return state


def transfer(x,T,outcomes,state_Q,log_Z,state_zero,L,debug=False):
    """_summary_

    Args:
        x (_type_): transfer matrix is applied on qubit x, x+1
        T (_type_): Transfer matrix
        outcomes (_type_): measurement outcomes at x, x+1. outcome has 3 possible values, 0: no measurement, 1: measured outcome is 1, -1: measured outcome is 0
        state_Q (_type_): state of the system
        log_Z (_type_): list of log of the partition functions in the previous steps
        state_zero (Boolean): if true then state_Q has zero partition function/weight; the measurement outcomes are compatible with the total charge.
        L (_type_): system size
        debug (bool, optional): Defaults to False.

    Returns:
        state_Q: transferred state
        state_zero: if true, the partition function becomes zero
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


def sep_dynamics_2(data,Q,neel_initial_state=True):
    """_summary_

    Args:
        data (_type_): 2d array of shape (depth,L) holding values of the outcomes
        Q (_type_): initial charge of the quantum state which was used to generate the outcomes
        neel_initial_state (bool, optional): Defaults to True.

    Returns:
        _type_: _description_
    """
    (depth,L) = data.shape
   
    """
    Q2 = Q-1
    if Q<L//2:
        Q2 = Q+1
    """
    # assuming charge L//2 or L//2 + 1 
    Q2 = Q + 1
    if Q > L//2:
        Q2 = Q - 1
    total = 0
    p_success = []
    
    initial_state_Q = initial_state(L,Q,neel_state=neel_initial_state)
    initial_state_Q2 = initial_state(L,Q2,neel_state=neel_initial_state)

    N=1
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
    total += N
    for t in range(depth)[:]:
        
        if t%2 == 0: #even layer
            for x in range(0,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
            
                state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L)
                """ 
                # debug step
                if state_Q_is_zero and state_Q2_is_zero:
                    print('hurray2',np.sum(state_Q2),np.sum(state_Q))
                """    
        else:
            for x in range(1,L-1,2):
                outcomes = (traj[t,x],traj[t,x+1])
                debug=False

                state_Q, state_Q_is_zero = transfer(x,T,outcomes,state_Q,log_Z,state_Q_is_zero,L,debug=False)

                state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L)
                """
                # debug step
                if state_Q_is_zero and state_Q2_is_zero:
                    print('hurray2',np.sum(state_Q2),np.sum(state_Q))
                """

        # print(t,time.time()-start,total)
        if state_Q_is_zero:              
            break
    
        if state_Q2_is_zero:
            break
        
    if state_Q_is_zero:
        p_Q = 0
    elif state_Q2_is_zero:
        p_Q = 1
    else:
        ratio_p = np.exp(np.sum(log_Z) - np.sum(log_Z2)) # ratio of the prob. is the ratio of the respective partition function in the SEP dynamics
        p_Q = 1/(1+1/ratio_p)
        p_Q2 = 1/(1+ratio_p)
    
    return p_Q



def get_sep_data(Q,L,p,depth_ratio,scrambled):
    depth = int(L*depth_ratio)
    p_suc = []
    seed=2
    file_dir = 'data/measurement_data_fixed/seed='+str(seed)
    neel_state = True
    if scrambled:
        neel_state = False
        filename = 'data/measurement_data_fixed_scrambled/seed='+str(seed)
    
    filename = file_dir +'/L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p=' + str(p)+ '_seed='+str(seed)
    with open(filename,'rb') as f:
        data_raw,_,_ = pickle.load(f)

    for data in data_raw:
        p_suc.extend([sep_dynamics_2(data[0],Q,neel_initial_state=neel_state)]*data[1])
    return p_suc

"""
p_list = [0.05,0.1,0.13,0.16,0.2,0.25,0.3,0.4]
L_list = [6,8,10,12,14]
p_suc_dic = {}
scrambled=False
final_file = 'sep_data/seed=2_fixed'
if scrambled:
    final_file = final_file + '_scrambled'
for L in L_list:
    print(L)
    p_suc_dic[L] = {}
    for p in p_list:
        p_suc_dic[L][p] = {}
        for Q in [L//2,L//2-1]:
            start = time.time()
            p_suc_dic[L][p][Q] = get_sep_data(Q,L,p,depth_ratio=1,scrambled=scrambled)
            print(L,p,Q,time.time()-start)
    with open(final_file,'wb') as f:
        pickle.dump(p_suc_dic,f)
"""
here = os.path.dirname(os.path.abspath(__file__))
accuracy_array = []
for circuit_iter in range(1,11):
    try:
        filename = os.path.join(here, "data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,0,number_shots,circuit_iter))
        measurement_record_0 = np.load(filename)
        filename = os.path.join(here, "data/measurement_record_L_{}_p_{}_Q_{}_numbershots_{}_iter_{}.npy".format(L,p,1,number_shots,circuit_iter))
        measurement_record_1 = np.load(filename)        
        measurement_records = np.concatenate([measurement_record_0,measurement_record_1],axis=0)
        num_meas_records_0 = len(measurement_record_0[:,0,0])
        num_meas_records_1 = len(measurement_record_1[:,0,0])   
        num_meas_records = num_meas_records_0+num_meas_records_1
        permut = np.random.permutation(num_meas_records) 
        data = measurement_records[permut,:,:]
        charge_output_0 = np.zeros(num_meas_records_0)
        charge_output_1 = np.ones(num_meas_records_1)
        charge_output = np.concatenate([charge_output_0,charge_output_1],axis=0)
        labels = charge_output[permut]
        test_percentage = 0.2 
        train_percentage = 1 - test_percentage 
        number_samples = len(measurement_records)
        test_data_number_samples = round(test_percentage * number_samples)
        test_data = data[0:test_data_number_samples,:,:]
        num_different_records = len(measurement_record_0[:,0,0])
        accuracy = []
        for i in range(0,test_data_number_samples):
            charge = int(L//2 + labels[i])
            proba_success = sep_dynamics_2(test_data[i,:,:], charge)
            accuracy.append(proba_success > 0.5)
        accuracy_array.append(np.mean(accuracy))
    except:
        print("ignoring circuit iter", circuit_iter)
np.save("data/test_accuracy_SEP_L_{}_p_{}_numbershots_{}.npy".format(L,p,number_shots), accuracy_array) # store all measurements except for final measurements 