"""
Use the "quantum jump" method to simulate a laser driven TLE
Compute the first order correlation function and obtain the Mollow triplet
"""
import numpy as np
from numpy import linalg as npLA# multi_dot
from matplotlib import pyplot as plt
import time

time_start = time.time()
'''
########################   DEFINE BASIC PARAMETERS   ##########################
'''
Z_1 = 400 #500                       # number of realisations of different \psi
Z_2 = 5 #5                               # number of trajectories for each \psi

delta = 0.                                             # emitter-laser detuning
eta = 6                                          # laser power (Rabi frequency)
gamma = 1.                                                      # coupling rate

N = 2                                                 # Hilbert space dimension
'''
#############################   DEFINE OPERATORS   ############################
'''
sig = np.zeros([N,N])                            # initialize lowering operator
sqrt_n_vec = np.sqrt(range(1,N))
np.fill_diagonal(sig[:,1:],sqrt_n_vec)             # complete lowering operator

sig_dag = sig.conj().transpose()                             # raising operator

A = sig_dag                         # A-operator in the correlation calculation
B = sig                             # B-operator in the correlation calculation

N_op = sig_dag.dot(sig)                            # excitation number operator

n_th = 0.0                       # average photon number in thermal equilibrium

C_op_1 = np.sqrt(gamma*(1+n_th))*sig                  # jump operator (destroy)
C_op_1_dag = C_op_1.conj().transpose()                           # h.c. of -||-
C_op_2 = np.sqrt(gamma*(n_th))*sig_dag                 # jump operator (create)
C_op_2_dag = C_op_2.conj().transpose()                           # h.c. of -||-

H = delta*sig_dag.dot(sig) + eta/2*(sig+sig_dag)                  # hamiltonian
H_eff = H - 1j/2*C_op_1_dag.dot(C_op_1) \
        - 1j/2*C_op_2_dag.dot(C_op_2)                   # effective hamiltonian
expm_arg = -1j*H_eff
alpha, beta = np.roots([1,-(expm_arg[0,0]+expm_arg[1,1]), \
           -expm_arg[1,0]*expm_arg[0,1]]) # for calculating the matrix exponent
'''
###########################   SIMULATION PARAMETERS   #########################
'''
### time parameters for reaching steady state ###
t0 = 0.                                                          # initial time
t_end = 2*gamma+1.5*eta                                              # end time
t_steps = 250                                            # number of time steps 
t_vec = np.linspace(t0,t_end,t_steps)                             # time vector
dt = t_vec[1]-t_vec[0]                                         # time increment
### time parameters for calculating the correlation function ###
t0_2 = 0.                                                        # initial time
t_end_2 = 3*gamma +1.5*eta                                           # end time
t_steps_2 = 250                                          # number of time steps 
t_vec_2 = np.linspace(t0_2,t_end_2,t_steps_2)                     # time vector
dt_2 = t_vec_2[1]-t_vec_2[0]                                   # time increment

ev_N_mat = np.zeros([Z_1,np.size(t_vec)])           # preallocate photon number 
### preallocate partial correlation varible
c_plus_mat = 1j*np.zeros([Z_2,Z_1,np.size(t_vec_2)])      
c_minus_mat = 1j*np.zeros([Z_2,Z_1,np.size(t_vec_2)])      
c_prime_plus_mat = 1j*np.zeros([Z_2,Z_1,np.size(t_vec_2)])     
c_prime_minus_mat = 1j*np.zeros([Z_2,Z_1,np.size(t_vec_2)])      
'''
###############################   FUNCTIONS   #################################
'''
def psi_exists(psi_in):
    """
    'psi_exists' determines if the input state exists.
    
    ... i.e. the function determines whether or not the state ('psi_in') can be
    normalised, i.e. if the input is different from the zero-vector 
    (or practically zero)
    """
    psi_in_dual = psi_in.conj().transpose()         # input state in dual space
    psi_out_braket = np.matmul(psi_in_dual,psi_in )             # inner product
    threshold_overlap = 1e-5
    
    if np.absolute(psi_out_braket)[0,0] > threshold_overlap:         # criteria
        psi_exists_out = True     # return 'True' is the overlap is significant
    else:
        psi_exists_out = False                   # ... otherwise return 'False'
        print('error')
    return psi_exists_out

def is_jump(R,psi_in,t):
    """
    'is_jump' determines if the state should make a jump.
    
    The evolution of the input state wrt. the effective Hamiltonian is compared
    to the random number r in order to determine whether or not a jump occurs.
    """
    s0 = (alpha*np.exp(beta*t)-beta*np.exp(alpha*t))/(alpha-beta)
    s1 = (np.exp(alpha*t)-np.exp(beta*t))/(alpha-beta)
    U = s0*np.eye(2)+s1*expm_arg                 # effective evolution operator
    psi_bar = U.dot(psi_in);                                    # evolved state
    psi_bar_dual = psi_bar.conj().transpose()     # hermitian jonjugate of -||-
    braket_psi_bar = np.matmul(psi_bar_dual,psi_bar)            # \braket{\psi}

    if braket_psi_bar < R:
        jump_is = True                         # return 'True' if a jump occurs
    else:
        jump_is = False                               # ... else return 'False'
    return jump_is

def psi_nj(psi_in,DT):
    """
    'psi_nj' determines the normalized state if no jump occurs.
    
    If no jump occurs the state is evolved according to the effective 
    Hamiltonian and then normalized.
    """
    if psi_exists(psi_in):                            # If the input exists ...
        #U_dt = spLA.expm(-1j*DT*H_eff)  
        s0 = (alpha*np.exp(beta*DT)-beta*np.exp(alpha*DT))/(alpha-beta)
        s1 = (np.exp(alpha*DT)-np.exp(beta*DT))/(alpha-beta)
        U_dt = s0*np.eye(2)+s1*expm_arg  # ... determine evolution operator ...
        psi_out_bar = U_dt.dot(psi_in);           # ... and calculate the state
        psi_out_bar_dual = psi_out_bar.conj().transpose()
        psi_out_braket = np.matmul(psi_out_bar_dual,psi_out_bar)
        psi_out = psi_out_bar/np.sqrt(psi_out_braket)     # normalise the state
    else:                                    # If the state does not exists ...
        psi_out = np.zeros([N,1])      # ... let the state remain a zero-vector
    return psi_out

def psi_j(psi_in):
    """
    'psi_j' determines the normalized state if a jump occurs.
    
    If a jump occurs the jump operator should be applied on the state and 
    succedingly normalized.
    """
    if psi_exists(psi_in):                            # If the state exists ...
        # ... the state will be either
        psi_out_1_bar = C_op_1.dot(psi_in)
        psi_out_1_bar_dual = psi_out_1_bar.conj().transpose()
        psi_out_1_braket = np.matmul(psi_out_1_bar_dual , psi_out_1_bar)
        # ... or
        psi_out_2_bar = C_op_2.dot(psi_in);
        psi_out_2_bar_dual = psi_out_2_bar.conj().transpose()
        psi_out_2_braket = np.matmul(psi_out_2_bar_dual , psi_out_2_bar)
        
        P_1 = np.real(psi_out_1_braket/(psi_out_1_braket+psi_out_2_braket))
        # ... we decide here
        r_2 = np.random.uniform(size = 1)
        if P_1 > r_2:
            psi_out = psi_out_1_bar/np.sqrt(psi_out_1_braket)
        else:
            psi_out = psi_out_2_bar/np.sqrt(psi_out_2_braket)
    else:                                    # If the input does not exists ...
        psi_out = np.zeros([N,1])      # ... let the state remain a zero-vector
    return psi_out

'''
#################################   SIMULATE   ################################
'''
for z1 in range(Z_1):                                           # loop over Z_1
    ket_N = 0                                           # initial photon number
    psi_0 = np.zeros([N,1]); psi_0[ket_N,0] = 1.                # initial state
    psi_0_dual = psi_0.conj().transpose()                  # dual space of -||-
    psi = psi_0;    psi_dual = psi.conj().transpose() 
        
    r1 = np.random.uniform(size = 1)                            # random number
    tau = 0                                                    # temporary time
    for tt in range(np.size(t_vec)):                      # loop over all times  
        ev_N_mat[z1,tt] = np.real(npLA.multi_dot( [ psi_dual , \
                         N_op , psi ] )[0,0])            # update photon number
                        
        has_jumped = is_jump(r1,psi_0,tau)         # has the state made a jump?
                
        if has_jumped:                            # if the state has jumped ...
            psi = psi_j(psi)                     # apply the jump operation ...
            psi_0 = psi
            tau = 0.
            r1 = np.random.uniform(size = 1)          # and reset the algorithm       
        else:                                 # if the state has not jumped ...
            psi = psi_nj(psi,dt)            # apply the "no jump" operation ...
            tau = tau + dt                                    # .. and continue
                
        psi_dual = psi.conj().transpose()        
    
    for z2 in range(Z_2):                                       # loop over Z_2       
        '''
        #############################   INITIALIZE CHIS   #####################
        '''
        ### initial states (not normalized) ###
        chi_plus_0_NN = np.dot( ( np.eye(2) + B )  , psi )          
        chi_minus_0_NN = np.dot( ( np.eye(2) - B )  , psi )         
        chi_prime_plus_0_NN = np.dot( ( np.eye(2) +1j* B )  , psi )  
        chi_prime_minus_0_NN = np.dot( ( np.eye(2) -1j* B )  , psi )  
        ### dual space representation of -||- ###
        chi_plus_0_NN_dual = chi_plus_0_NN.conj().transpose()                  
        chi_minus_0_NN_dual = chi_minus_0_NN.conj().transpose()                 
        chi_prime_plus_0_NN_dual = chi_prime_plus_0_NN.conj().transpose()                  
        chi_prime_minus_0_NN_dual = chi_prime_minus_0_NN.conj().transpose()  
        ### normalisation constant ###
        mu_plus = np.matmul(chi_plus_0_NN_dual , chi_plus_0_NN)
        mu_minus = np.matmul(chi_minus_0_NN_dual , chi_minus_0_NN)
        mu_prime_plus = np.matmul(chi_prime_plus_0_NN_dual , \
                                  chi_prime_plus_0_NN)
        mu_prime_minus = np.matmul(chi_prime_minus_0_NN_dual , \
                                   chi_prime_minus_0_NN)
        ### normalised initial states ###
        chi_plus_0 = chi_plus_0_NN/np.sqrt(mu_plus)
        chi_minus_0 = chi_minus_0_NN/np.sqrt(mu_minus)
        chi_prime_plus_0 = chi_prime_plus_0_NN/np.sqrt(mu_prime_plus)
        chi_prime_minus_0 = chi_prime_minus_0_NN/np.sqrt(mu_prime_minus)
        ### dual space representation of -||- ###
        chi_plus_0_dual = chi_plus_0.conj().transpose()
        chi_minus_0_dual = chi_minus_0.conj().transpose()
        chi_prime_plus_0_dual = chi_prime_plus_0.conj().transpose()
        chi_prime_minus_0_dual = chi_prime_minus_0.conj().transpose()
        ### initial states again ###
        chi_plus = chi_plus_0;
        chi_minus = chi_minus_0;    
        chi_prime_plus = chi_prime_plus_0;    
        chi_prime_minus = chi_prime_minus_0;    
        ### dual space representation of -||- ###   
        chi_plus_dual = chi_plus.conj().transpose() 
        chi_minus_dual = chi_minus.conj().transpose()  
        chi_prime_plus_dual = chi_prime_plus.conj().transpose() 
        chi_prime_minus_dual = chi_prime_minus.conj().transpose() 
        '''
        #############################   SIMULATE CHI PLUS   ###################
        '''
        r2_a = np.random.uniform(size = 1)                      # random number
        tau = 0                                                # temporary time
        for ttt in range(np.size(t_vec_2)):               # loop over all times
            c_plus_mat[z2,z1,ttt] = npLA.multi_dot( [ chi_plus_dual , \
                        A , chi_plus ] )[0,0]      # update partial correlation
                
            has_jumped = is_jump(r2_a,chi_plus_0,tau)   # has the state jumped?
                      
            if has_jumped:                        # if the state has jumped ...
                chi_plus = psi_j(chi_plus)       # apply the jump operation ...
                chi_plus_0 = chi_plus
                tau = 0.
                r2_a = np.random.uniform(size = 1)    # and reset the algorithm
            else:                             # if the state has not jumped ...
                chi_plus = psi_nj(chi_plus,dt_2) # apply "no jump" operation...
                tau = tau + dt_2                                 # and continue
                
            chi_plus_dual = chi_plus.conj().transpose()   
        '''
        #############################   SIMULATE CHI MINUS   ##################
        '''          
        ### see commented code above ('SIMULATE CHI PLUS') ###
        r2_b = np.random.uniform(size = 1)                         
        tau = 0                                                    
        for ttt in range(np.size(t_vec_2)):
            c_minus_mat[z2,z1,ttt] = npLA.multi_dot( [ chi_minus_dual , \
                              A , chi_minus ] )[0,0]                    
                        
            has_jumped = is_jump(r2_b,chi_minus_0,tau)         
                
            if has_jumped:                            
                chi_minus = psi_j(chi_minus)                   
                chi_minus_0 = chi_minus
                tau = 0.
                r2_b = np.random.uniform(size = 1)           
            else:                                 
                chi_minus = psi_nj(chi_minus,dt_2)               
                tau = tau + dt_2                                      
                
            chi_minus_dual = chi_minus.conj().transpose()
        '''
        ######################   SIMULATE CHI PRIME PLUS   ####################
        '''
        ### see commented code above ('SIMULATE CHI PLUS') ###
        r2_c = np.random.uniform(size = 1)                         
        tau = 0                                                  
        for ttt in range(np.size(t_vec_2)):                      
            c_prime_plus_mat[z2,z1,ttt] = npLA.multi_dot( [ \
            chi_prime_plus_dual , A , chi_prime_plus ] )[0,0]                   
                        
            has_jumped = is_jump(r2_c,chi_prime_plus_0,tau)         
                
            if has_jumped:                          
                chi_prime_plus = psi_j(chi_prime_plus)                   
                chi_prime_plus_0 = chi_prime_plus
                tau = 0.
                r2_c = np.random.uniform(size = 1)           
            else:                                 
                chi_prime_plus = psi_nj(chi_prime_plus,dt_2)  
                tau = tau + dt_2                                    
                
            chi_prime_plus_dual = chi_prime_plus.conj().transpose() 
        '''
        ######################   SIMULATE CHI PRIME MINUS   ###################
        '''
        ### see commented code above ('SIMULATE CHI PLUS') ###
        r2_d = np.random.uniform(size = 1)                            
        tau = 0                                                    
        for ttt in range(np.size(t_vec_2)):                      
            c_prime_minus_mat[z2,z1,ttt] = npLA.multi_dot( [ \
                    chi_prime_minus_dual ,  A , chi_prime_minus ] )[0,0]                    
                        
            has_jumped = is_jump(r2_d,chi_prime_minus_0,tau)        
                
            if has_jumped:                         
                chi_prime_minus = psi_j(chi_prime_minus)                    
                chi_prime_minus_0 = chi_prime_minus
                tau = 0.
                r2_d = np.random.uniform(size = 1)          
            else:                                 
                chi_prime_minus = psi_nj(chi_prime_minus,dt_2)             
                tau = tau + dt_2                                       
                
            chi_prime_minus_dual = chi_prime_minus.conj().transpose() 
'''
############################   ENSEMBLE AVERAGE  ##############################
'''
ev_N_vec = np.mean(ev_N_mat,axis=0)             # average over all realisations
'''
########################   CORRELATION FUNCTION  ##############################
'''         
### take the ensemble average ###
c_plus_bar = np.mean(np.mean(c_plus_mat,axis=1),axis=0)
c_minus_bar = np.mean(np.mean(c_minus_mat,axis=1), axis = 0)
c_prime_plus_bar = np.mean(np.mean(c_prime_plus_mat,axis=1), axis = 0)
c_prime_minus_bar = np.mean(np.mean(c_prime_minus_mat,axis=1), axis = 0)
C_vec = 1/4 * ( mu_plus * c_plus_bar - mu_minus * c_minus_bar \
            -1j*mu_prime_plus * c_prime_plus_bar \
            +1j*mu_prime_minus * c_prime_minus_bar)        # correlation vector
C_vec = C_vec.conj().transpose()
C_abs_vec = np.abs(C_vec)                              # absolute value of -||-

### fourier transform ###
w_vec = np.arange(-15,15,0.05);
tt, ww = np.meshgrid( t_vec_2 , w_vec)
FT_mat = np.exp(1j*ww*tt)
C_FT = dt_2 * np.dot(FT_mat , C_vec )
C_FT_abs = np.abs(C_FT)

time_end = time.time()

print(['Time ellapsed: ', time_end-time_start,' s'])
'''
##################################   PLOTS ####################################
'''
plt.rc('text',usetex=True)
plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.size"]=14
plt.rc('text.latex', preamble= \
      r'\usepackage{amsmath} \usepackage{mathptmx} \usepackage{newtxmath}')

##############################################################################
fig = plt.figure(figsize=(3.4,3.0))
fig.set_size_inches(3.4, 3)
plt.plot(t_vec,ev_N_vec,'k-')
plt.xlabel('time, $t\gamma$')
plt.ylabel(r'$\langle \sigma^{\dagger}(t )\sigma(t) \rangle $')
plt.xlim((0,t_end))
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.20)
plt.show()
# save figure to .pdf
fig.savefig('excitation_number.pdf')

###############################################################################
fig= plt.figure(figsize=(3.4,3.0))
fig.set_size_inches(3.4, 3)
plt.plot(t_vec_2,C_abs_vec,'k-')
plt.xlabel(r'time, $\tau\gamma$')
plt.ylabel(r'$\langle \sigma^{\dagger}(t_1+ \tau )\sigma(t_1) \rangle $')
plt.xlim((0,t_end_2))
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.20)
plt.show()
# save figure to .pdf
fig.savefig('temporal_correlation.pdf')

###############################################################################
fig = plt.figure(figsize=(3.4,3.0))
plt.plot(w_vec,C_FT_abs,'k-')
plt.xlabel('frequency, $(\omega-\omega_\mathrm{e})/\gamma$')
plt.ylabel('$S(\omega) \, [a.u.]$')
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.20)
plt.show()
# save figure to .pdf
fig.savefig('Mollow_triplet.pdf')