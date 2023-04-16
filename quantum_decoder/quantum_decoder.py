import time
import numpy as np
import pickle
import circuit_numpy_utils
from itertools import combinations

def initial_state(L):
    state = np.zeros((2,)*L)
    Q_state = [1,0]*(L//2)
    Q2_state = [1,0]*(L//2 - 1) + [0,0]
    state[tuple(Q_state)] = 1/2**0.5
    state[tuple(Q2_state)] = 1/2**0.5

    return state

def unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_zero,L,theta,do_measure,debug=False):
    """_summary_

    Args:
        x,y(_type_): transfer matrix is applied on qubit x, y

        T (_type_): Transfer matrix
        outcomes (_type_): measurement outcomes at x, x+1. outcome has 3 possible values, 0: no measurement, 1: measured outcome is 1, -1: measured outcome is 0
        state_Q (_type_): state of the system
        log_Z (_type_): list of log of the partition functions in the previous steps
        state_zero (Boolean): if true then state_Q has zero partition function/weight; the measurement outcomes are compatible with the total charge.
        L (_type_): system size
        theta: quantify the strength of the weak measurement. theta=pi/2 is proj. measurement
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

        if outcomes[0] == 1:
            state_Q[(0,1),:] = 0
            state_Q[(2,3),:] = -1j*np.sin(theta) * state_Q[(2,3),:]
        elif outcomes[0] == -1:
            state_Q[(2,3),:] = np.cos(theta) *state_Q[(2,3),:]

        if outcomes[1] == 1:
            state_Q[(0,2),:] = 0
            state_Q[(1,3),:] = -1j * np.sin(theta) * state_Q[(1,3),:]
        elif outcomes[1] == -1:
            state_Q[(1,3),:] = np.cos(theta) *state_Q[(1,3),:]
        
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
    count = 0
    p_list = []
    for positions in combinations(range(L), Q):
            p = [0] * L

            for i in positions:
                p[i] = 1
            count += 1
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


def quantum_dynamics_2(data,Q,theta,U_list, decoding_protocol=0):
    """
    input:
        - data (_type_): 2d array of shape (depth,L) holding values of the outcomes
        - Q (_type_): initial charge of the quantum state which was used to generate the outcomes
        - neel_initial_state (bool, optional): Defaults to True. This argument is redundant now!
        - decoding_protocol
            0: No active decoding. Post-select such that both P_Q and P_Q1 are 0
            1: Postselect on trajectories where P_suc != 0, i.e P_Q != 0
            2: Postselect on trajectories where last layer has total charge = Q
            3: Union of 2 and 1

    Returns:
        p_Q: Probility of the initial charge being Q (the true charge) in the SEP dynamics
    """

    (depth,L) = data.shape
    
    Q2 = Q-1
    if Q<L//2:
        Q2 = Q+1

    indices_Q = get_indices(L,Q)
    indices_Q2 = get_indices(L,Q2)

    if decoding_protocol == 2 or decoding_protocol == 3:
        if np.sum((data[-1,:]+1)) != 2*Q:
            print('Alert!',L,Q,theta)
            print(data[:,:],Q)
            return False

    total = 0
    p_success = []
    
    state = initial_state(L)

    N=1
    traj = data
    state_Q_is_zero = False

    # state_Q2_is_zero = False
    # state_Q = initial_state_Q.copy()
    # state_Q2 = initial_state_Q2.copy()

    T = len(U_list) # includes t_scr and depth

    log_Z = []
    log_Z2 = []
    total += N
    for t in range(T)[:-1]:
            for x,y,U in U_list[t]:
                if t >= T-depth:
                    outcomes = (traj[t-T+depth,x],traj[t-T+depth,y])
            
                    state, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state,log_Z,state_Q_is_zero,L,theta,do_measure=True,debug=False)
                
                else: #scrambling step
                    outcomes = None
                    state, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state,log_Z,state_Q_is_zero,L,theta,do_measure=False,debug=False)

                p_Q = get_probability(state,L,Q,indices=indices_Q)
                p_Q2 = get_probability(state,L,Q2,indices=indices_Q2)

                # assert np.round(p_Q + p_Q2,8) == 1, (p_Q,p_Q2,np.round(p_Q+p_Q2,8))

                # state_Q2, state_Q2_is_zero = transfer(x,T,outcomes,state_Q2,log_Z2,state_Q2_is_zero,L,theta)
                

                if p_Q == 0:
                    if decoding_protocol == 1 or decoding_protocol == 3:
                        return False
                    if p_Q2 == 0:
                        return False

            p_success.append(p_Q)
    
    return p_success


def get_data(Q,L,p,depth_ratio,scrambling_type,is_noisy,decoding_protocol=0):
    depth = int(L*depth_ratio)
    p_suc = []
    seed=1
    U_list = circuit_numpy_utils.get_circuit(L,int(depth_ratio*L),scrambling_type=scrambling_type,seed=seed,t_scram=5)
    if depth_ratio != 1:
        depth_label= "_depth_ratio="+str(depth_ratio)
    else:
        depth_label = ""

    if is_noisy:
        noisy_label = '_noisy'
    else:
        noisy_label = ''

    if scrambling_type is None:
        scrambling_label = '_no_scrambling'
        noisy_label = ''
    elif scrambling_type == 'Normal':
        scrambling_label = '_normal'
    elif scrambling_type == 'Special':
        scrambling_label = '_special'
    else:
        print("Scrambled input argument not recognized. It should be either \'Normal\', \'Special\' or None")
        return

    filedir = 'Weak measurements/data/qiskit_data/measurement_data_all_qubits'+ scrambling_label + noisy_label + depth_label+'/'
    
    filename = filedir +'/L='+str(L)+'_depth='+str(depth)+'_Q='+str(Q)+'_p=' + str(p)+ '_seed='+str(seed)
    with open(filename,'rb') as f:
        data_raw,_,_ = pickle.load(f)

    faulty_traj = 0
    total_traj = 0
    for data in data_raw:
        result = quantum_dynamics_2(data[0],Q,p,U_list=U_list,decoding_protocol=decoding_protocol)
        if result is False:
            faulty_traj += data[1]
        else:
            p_suc.extend([np.array(result)]*data[1])
        total_traj += data[1]

    return p_suc,faulty_traj/total_traj


def run():
    p_list = np.round(np.linspace(0,np.pi/2,10),3)[:]
    p_list = np.round(np.linspace(0.2,0.6*np.pi/2,15),3)[:]
    p_list = np.round(np.linspace(0,0.2,5),3)

    L_list = [6,8,10,12,14,16][:4]
    p_suc_dic = {}
    scrambling_type = 'Special'
    depth_ratio=1

    for is_noisy in [False]:
        for decoding_protocol in [3]:
            final_file = 'Weak measurements/data_quantum_decoder/decoder/seed=1_all_qubits'
            if scrambling_type == 'Special':
                final_file = final_file + '_special_scrambled'
            elif scrambling_type == 'Normal':
                final_file = final_file + '_normal_scrambled'

            final_file = final_file + '_decoding_protocol='+str(decoding_protocol)

            if is_noisy:
                final_file = final_file + '_noisy'

            with open(final_file,'rb') as f:
                p_suc_dic = pickle.load(f)

            for L in L_list:
                if L not in p_suc_dic:
                    p_suc_dic[L] = {}
                for p in p_list:
                    p_suc_dic[L][p] = {}
                    for Q in [L//2,L//2-1][:]:
                        start = time.time()
                        p_suc_dic[L][p][Q],temp = get_data(Q,L,p,depth_ratio=depth_ratio,scrambling_type=scrambling_type,is_noisy=is_noisy,decoding_protocol=decoding_protocol)
                        print(L,p,Q," frac of faulty traj:",temp," time=",time.time()-start,'\n',"is_noisy:",is_noisy,' decoding_protocol:',decoding_protocol)
                with open(final_file,'wb') as f:
                    pickle.dump(p_suc_dic,f)

# run()