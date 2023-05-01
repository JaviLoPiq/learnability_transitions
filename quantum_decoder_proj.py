import time
import numpy as np
import pickle
from itertools import combinations
import qiskit as qk
import sys
#sys.path.insert(1, '/Users/javier/Dropbox/Projects/measurement transitions/learnability_transitions') # TODO: import all files needed
from U1MRC import unitary_gate_from_params, U1MRC
circuit_reali = 10

def initial_state(L):
    state = np.zeros((2,)*L)
    Q_state = [1,0]*(L//2)
    """
    # assuming Q2 = L//2 -1
    Q2_state = [1,0]*(L//2 - 1) + [0,0]
    """
    # assuming Q2 = L//2 + 1
    Q2_state = [1,0]*(L//2 - 1) + [1,1]
    state[tuple(Q_state)] = 1/2**0.5
    state[tuple(Q2_state)] = 1/2**0.5

    return state

def unitary_measurement(x,y,U,outcomes,state_Q,log_Z,state_zero,L,do_measure,debug=False):
    """_summary_

    Args:
        x,y(_type_): transfer matrix is applied on qubit x, y

        T (_type_): Transfer matrix
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


def quantum_dynamics_2(data,Q,U_list, decoding_protocol=0):
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
    indices_Q2 = get_indices(L,Q2)

    if decoding_protocol == 2 or decoding_protocol == 3:
        if np.sum((data[-1,:]+1)) != 2*Q:
            print('Alert!',L,Q)
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
                if t >= T-depth: # TODO: or t >= depth?
                    outcomes = (traj[t-T+depth,x],traj[t-T+depth,y])

                    state, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state,log_Z,state_Q_is_zero,L,do_measure=True,debug=False)

                else: #scrambling step
                    outcomes = None
                    state, state_Q_is_zero = unitary_measurement(x,y,U,outcomes,state,log_Z,state_Q_is_zero,L,do_measure=False,debug=False)

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

"""
def get_data(Q,L,p,depth_ratio,scrambling_type,is_noisy,decoding_protocol=0):
    depth = int(L*depth_ratio)
    p_suc = []
    seed=1
    u1mrc = U1MRC(number_qubits=L, depth=depth, measurement_locs=m_locs, params=param_list, initial_charge=initial_charge, debug=False)
    U_list, = U1MRC.generate_u1mrc(L,int(depth_ratio*L),scrambling_type=scrambling_type,seed=seed,t_scram=5)
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
"""
# run()
L = 10
depth = L
PARAMS_PER_GATE = 6
p = int(sys.argv[1])/10
number_shots = 500
depth_ratio = 1
scrambling_type = 'Special'
is_noisy = False
decoding_protocol = 0
m_locs = np.random.binomial(1,p,L*(depth-1)).reshape((depth-1,L))

# generate random circuit parameters
# each layer has L//2
#params = 4*np.pi*np.random.rand(depth,L//2,PARAMS_PER_GATE)
param_list = [[4*np.pi*np.random.rand(PARAMS_PER_GATE) 
            for j in range(L//2-(i%2))] # there are either L//2 or L//2-1 gates for even/odd layers resp.
            for i in range(depth)]

accuracy_list = []
for s in range(0,2):
    Q = L//2 + s
    #quantum_dynamics_2(Q,L,p,depth_ratio=depth_ratio,scrambling_type=scrambling_type,is_noisy=is_noisy,decoding_protocol=decoding_protocol)
    u1mrc = U1MRC(number_qubits=L, depth=depth, measurement_locs=m_locs, params=param_list, initial_charge=Q, debug=False)
    circ, U_list = u1mrc.generate_u1mrc(measurement_rate=p, reali=circuit_reali, save_state=False, save_unitaries=True)
    backend = qk.Aer.get_backend('qasm_simulator')
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

    for i in range(number_different_outcomes):
        accuracy_list.append(quantum_dynamics_2(measurement_record[i,:,:],Q, U_list, decoding_protocol=decoding_protocol)[2*(depth-1)] > 0.5)

np.save("data/accuracy_quantum_decoder_L_{}_p_{}_numbershots_{}_iter_{}_decoding_type_0.npy".format(L,p,number_shots,circuit_reali), accuracy_list) # store all measurements except for final measurements 
