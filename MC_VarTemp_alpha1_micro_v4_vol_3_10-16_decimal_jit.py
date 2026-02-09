# Dynamic Monte Carlo Simulation of Acrylamide (AM)/Acrylic Acid (AAc) Copolymerization
# by Free Radical Batch Polymerization

# Importing Libraries

import datetime
print(datetime.datetime.now())
today = datetime.datetime.now().strftime('%d/%m/%Y')

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import UnivariateSpline
from numba import jit,njit
import pandas as pd
from numba import types
from scipy.signal import lfilter
import matplotlib.ticker as mtick
import math
import h5py             # Biblioteca para manipulação de arquivos HDF5

# Additional packages
import os
import psutil
from numba.typed import List

# Save specifications

run_type = 'decimal_jit/' # folder name based the type of method used (binary, decimal, or decimal_jit)
save_path = f'data/results/{run_type}' # Buindong the folder saving path based on the run type
note = '_tend_3000' # keep '' if no notes

# Memory usage tracking setting

global mem_log,tracking_step,next_track,init_real_time
tracking_step = 30 # step to record the memory usage in seconds (execution time, not reaction time)
next_track = 0.0
mem_log = pd.DataFrame(columns=["run_time_s", "reaction_time_s","memory_MB"]) # pandas log of memory usage


#Plot Specifications


axf = 16
al = 16
smo = 0.0001
a = 8
b = 6
ft = 16
plt.rcParams.update({'font.size':ft})



#Model Variables

#Compenents Properties

MW1,MW2,MWW =71.08,72.06,18.01  #AM AA WATER g/mol molecular weight
rho1,rho2,rhow = 1.13e3,1.05e3,1e3 #comonomer densities g/L + water density
rhoI = 1210 #initiator density g/L ou 1.21 g/cm^3
f = 0.8  #initiator efficiency


#Adiabatic variables


dH1  = 13.8 * 4184  #(J/mol) 16 and 13.8 kcal/mol for AA and AM at 74 C
dH2 = 16 * 4184
Cpw, Cp1, Cp2 = 75.38, 96, 145   #J/mol.K


#Monte Carlo Parameters


NA = 6.022e23         #1/mol
n = 7                 #number of species in the system
nr = 17               #number of Reactions
V = 3.e-16           #L --> Control volume


#Variables set by the user --> Unique variables to be modified


mc = 0.2    # Monomer fraction weight (include here M1 and M2)
f10 = 0.3     # Monomer 1 fraction of the weight %(include here only M1)
nt = 80    # Initial number of total moles [mol] (include here M1,M2 and Water)
wmI = 0.0022 #Initiator wt fraction
MMI = 271.19 #Initiator Molar Mass g/mol
#CI0 = 0.001  # Initiator concentration [mol/L]
tend = 3000 # s --> final reaction time (original value set to 3000)

wm10, wm20 = mc * f10 , mc * (1-f10)  # initial weight fraction of AM and AA
wmw = 1-mc-wmI  # weight fraction of water
wm1, wm2 = wm10, wm20             # mass  fraction  of  AM and AA on a polymer free basis
print(wm1,wm2,wm1+wm2,wmw)

mM10 = nt / (1/MW1 + (wm2/wm1)/MW2 + (wmw/wm1)/MWW + (wmI/wm1)/MMI) # Monomer 1 mass [g]
mM20 = (wm2/wm1)*mM10 # Monomer 2 mass [g]
mw   = (wmw/wm1)*mM10 # Water mass [g]
mI = (wmI/wm1)*mM10 # Initiator mass [g]

nM10 = mM10/MW1     # Initial moles Monomer 1  [mol]
nM20 = mM20/MW2 # Initial moles Monomer 2  [mol]
nI = mI/MMI # Initial moles Initiator  [mol]
nw = nt - nM10 - nM20 - nI  # Initial solvent  [mol]


print(mM10,mM20,mw,nM10,nM20,nw,nI)

VM1  = mM10/rho1 # Volume of monomer 1 [L]
VM2  = mM20/rho2 # Volume of monomer 2 [L]
VW   = mw/rhow  # Volume of water [L]
VI   = mI/rhoI # Volume of initiator [L]
Vt   = VM1+VM2+VW+VI # Volume of solution [L]
print(Vt)
m0 = mM10 + mM20 + mw +mI # total mass [g]
print(m0)


#Concentrations


C10 = nM10/Vt # initial concentration of M1 (mol/L)
C20 = nM20/Vt # initial concentration of M2 (mol/L)
CI0 = nI/Vt # initial concentration of I (mol/L)
CW = nw/Vt # [mol/L]
print(C10, C20, CW, CI0)


#Reactivity


rm1 = -1.37*wm10+2.07
rm2 = 1.27*wm20+0.13

T = 40+273.15


#Auxiliary Functions


def Averages(L):
    lambda2 = 0.0
    lambda1 = 0.0
    lambda0 = 0.0
    for i in range(len(L)):
        L[i] = np.float64(L[i])
        lambda2 = lambda2 + np.float64(i)**2*L[i]
        lambda1 = lambda1 + np.float64(i)*L[i]
        lambda0 = lambda0 + L[i]
    i_n = lambda1/lambda0
    i_w = lambda2/lambda1
    return i_n, i_w

def cld_original(D):
    D_sort = np.sort(D, axis=None)

    print(type(D_sort[0]))
    D_sorted = [np.int64(x) for x in range(0)]
    for i in range(len(D_sort)):
        D_sorted.append(D_sort[i].astype(np.int32))
    print(type(D_sorted[0]))

    final_D = np.bincount(D_sorted)
    length = np.linspace(0, len(final_D), len(final_D))
    plt.plot(length[1:],final_D[1:], color = 'blue')

    return final_D, length

@jit(nopython=True)
def new_chain(nn,bg,fim,length,final_D,delta):
    D_2 = np.zeros(nn)
    length_2 = np.zeros(nn)
    length_log = np.zeros(nn)
    for i in range(nn):
        for j in range (bg,fim):
            D_2[i] = D_2[i] + final_D[j]
            length_2[i] = length_2[i] + length[j]
        length_2[i] = length_2[i]/delta
        length_log[i] = np.log10(length_2[i])
        D_2[i] = D_2[i]*length_2[i]*length_2[i]*2.3026 # i2*L
        bg = bg + delta
        fim = fim + delta
    return D_2, length_2, length_log

def funcld(length,Dnorm,smo,lb,ub,n,b):

    #plot1
    plt.figure(0)
    plt.xlabel('$log_{10}(n)$',fontsize = axf)
    plt.ylabel('$w_n$',fontsize = axf)
    plt.plot(length,Dnorm,'b*')
    axes = plt.gca()
    #axes.set_xlim([xl,xu])
    #axes.set_ylim([0,1.5])
    #plt.legend([f'N = {N:.2e}'],frameon=False,fontsize = axf)

    # plot2
    plt.figure(1)

    a = 1
    yy = lfilter(b,a,Dnorm)
    plt.plot(length,Dnorm,'r*',linewidth = 1,markersize = 2)
    #plt.plot(length, yy, linewidth=2, linestyle="-", c="b")  # smooth by filter


    # plot3
    plt.figure(3)
    plt.plot(length, yy, linewidth=2, linestyle="-", c="b")  # smooth by filter
    plt.plot(length,Dnorm,'r*',linewidth = 1,markersize = 2)
    plt.xlabel('$log_{10}(n)$',fontsize = axf)
    plt.ylabel('$w_n$',fontsize = axf)

    #axes.set_ylim([0,0.00006])
    #plt.legend([f'$f_1$ = {f1}'],frameon=False,fontsize = axf)
    axes = plt.gca()
    axes.set_xlim([lb,ub])
    axes.set_ylim([0,1.5])
    plt.xlabel('$log_{10}(n)$',fontsize = axf)
    plt.ylabel('$w_n$',fontsize = axf)
    name = "MCCLD.tiff"
    plt.savefig(name,format='png',dpi = 600,bbox_inches='tight')

    return length,yy




# Numba doesn't cover all methods of Python lists, so we have to create some of them as functions


# Numba List doesn't support popping the last element, not .pop(i)
@jit(nopython=True)
def pop_nested_at_index(nested_list, index_to_remove):
    new_list = List()
    removed = nested_list[0] # so it inffers the type

    for i in range(len(nested_list)):
        if i == index_to_remove:
            removed = nested_list[i]
        else:
            new_list.append(nested_list[i])

    return removed, new_list

# Numba List doesn't support .reverse
@jit(nopython=True)
def reverse_int64_list(int64_list):
    new_list = List()

    for i in range(len(int64_list) - 1, -1, -1):
        new_list.append(int64_list[i])

    return new_list

# Numba List doesn't support .extend or merging lists by adding them together
@jit(nopython=True)
def extend_int64_list(int64_list1,int64_list2):

    for ele in int64_list2:
        int64_list1.append(ele)

    return int64_list1






# Helper function to record memory (obs.: psutil not compatible with numba)
def record_memory(run_time,reaction_time,mem_log):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    mem_log.loc[len(mem_log)] = [run_time, reaction_time, mem_mb]




# To jit the Monte Carlo function, I need to defne the nested lists outside of it and pass them as arguments
# Explicitly creating empty lists with the correct type
int_list_type = types.ListType(types.int64)

# using numba List() object that can be nested when using jit

# Each line is a list of int64 with the decimal representation in chunks
LM_chunks  = List.empty_list(int_list_type)  # comonomer sequence in the dead chain
P1M_chunks = List.empty_list(int_list_type)
P2M_chunks = List.empty_list(int_list_type)

# current chunk index and chunk length (previous chunk lengths are all 63) (list of lists)
LM_chunks_idx_length  = List.empty_list(int_list_type)
P1M_chunks_idx_length = List.empty_list(int_list_type)
P2M_chunks_idx_length = List.empty_list(int_list_type)

# Monte Carlo


@jit(nopython=True)
def MonteCarlo(t,f,X,R,M10,M20,tend,wm1,wm2,LM_chunks,P1M_chunks,P2M_chunks,LM_chunks_idx_length,P1M_chunks_idx_length,P2M_chunks_idx_length):
    #global mem_log,tracking_step,next_track,init_real_time

    P1 = [np.int64(x) for x in range(0)] # living chain that the last monomer inserted was M1
    P11 = [np.int64(x) for x in range(0)] # counts how many M1 belongs to P1
    P2 = [np.int64(x) for x in range(0)] # living chain that the last monomer inserted was M2
    P21 = [np.int64(x) for x in range(0)] # counts how many M1 belongs to P2
    L = [np.int64(x) for x in range(0)]   # dead chain
    L1 = [np.int64(x) for x in range(0)] # counts how many M1 belongs to L
    #LM  = [str(x) for x in range(0)]  # comonomer sequence in the dead chain
    #P1M = [str(x) for x in range(0)]
    #P2M = [str(x) for x in range(0)]








    T = 40 + 273.15       # K Temperature

    x1, x2, xt = 0, 0, 0
    M1 , M2 = X[2] , X[3]
    eps = 1e-10



    while t <= tend:
        '''
        current_real_time = time.time() - init_real_time

        # Memory usage - comment if using jit (incompatible function with numba)
        if current_real_time > next_track:
            record_memory(current_real_time,t,mem_log)
            next_track += tracking_step
        '''
        # reaction constants that change by T (wm10 and wm20 are initial values):
        kd = 9.24e14*np.exp(-14915/T)     # kd, Initiator decomposition (paper 5 & 6)
        ktc11 = 2e10*np.exp(-(1991+1477*wm10)/T)  # Termination by combination   (paper 6)
        # Combination
        ktc22 = 9.78e11*np.exp(-1860/T)*(1.56-1.77*wm20-1.2*wm20**2+2.43*wm20**3)*30**(-0.44)*10**(-4.5*0.16)
        ktc12 = (ktc11+ktc22)/2
        # Let's desconsider termination reactions
        ktd11 = 0
        ktd12 = 0
        ktd22 = 0

        # These constants change by the monomer mass fractions and T (wm1 and wm2):
        kp11 = 9.5e7*np.exp(-2189/T)*np.exp(-wm1*(0.0016*T+1.015))  # Monomer 1 propagation
        ki1  = kp11   # initiation with monomer 1
        kf11 = kp11*0.00118*np.exp(-1002/T)                         # transfer to monomer 1
        kp22 = 3.2e7*np.exp(-1564/T)*(0.11+0.89*np.exp(-3*wm2))      # monomer 2 propagation
        ki2  = kp22       # initiation with monomer 2
        kf22 = kp22*7.5e-5        # kp22*0.00118*np.exp(-1002/T) # transfer to monomer 2
        kp12 = kp11/rm1
        kp21 = kp22/rm2
        kf12 = kf22
        kf21 = kf11

        R[10] = kd * X[0]                               # rxn 1: Initiator decomposition
        R[8]  = ki1 * X[1] * X[2] / (V * NA)             # rxn 2: Initiation
        R[9]  = ki2 * X[1] * X[3] / (V * NA)             # rxn 3: Initiation
        R[0]  = kp11 * X[4] * X[2] / (V * NA)           # rxn 4: Propagation
        R[1]  = kp12 * X[4] * X[3] / (V * NA)           # rxn 5: Propagation
        R[2]  = kp21 * X[5] * X[2] / (V * NA)           # rxn 6: Propagation
        R[3]  = kp22 * X[5] * X[3] / (V * NA)           # rxn 7: Propagation
        R[11] = ktc11 * X[4] * (X[4]-1) / (V * NA * 2)  # rxn 8: Termination by combination
        R[12] = ktc12 * X[4] * X[5] / (V * NA)          # rxn 9: Termination by combination
        R[13] = ktc22 * X[5] * (X[5]-1) / (V * NA * 2)  # rxn 10: Termination by combination
        R[14] = ktd11 * X[4] * (X[4]-1) / (V * NA)      # rxn 11: Termination by disproportionation
        R[15] = ktd12 * X[4] * X[5] / (V * NA)          # rxn 12: Termination by disproportionation
        R[16] = ktd22 * X[5] * (X[5]-1) / (V * NA)      # rxn 13: Termination by disproportionation
        R[4]  = kf11 * X[4] * X[2] / (V * NA)           # rxn 14: Transfer to monomer
        R[5]  = kf21 * X[5] * X[2] / (V * NA)           # rxn 15: Transfer to monomer
        R[6]  = kf22 * X[5] * X[3] / (V * NA)           # rxn 16: Transfer to monomer
        R[7]  = kf12 * X[4] * X[3] / (V * NA)           # rxn 17: Transfer to monomer

        x1 = 1 - X[2]/M10    # conversion of monomer 1
        x2 = 1 - X[3]/M20    # conversion of monomer 2
        xt = 1 - (X[2] + X[3])/(M10 + M20)  # total conversion

        R0 = np.sum(R)
        dt = 1.0/(R0)*np.log(1.0/np.random.rand()) # delta t

        rnd = np.random.rand()*R0

        if rnd <= R[0]: # P1 + M1 --> P1(+1) propagation rxn 4
            i = int(np.random.rand()*X[4])
            P1[i] += 1 # living polymer grows in 1 unit
            P11[i] += 1 # updating the number of M1 in P1
            X[2] -= 1                            # 1 M1 less
            #P1M[i] = P1M[i]+'0' # M1 added


            # Adding M1
            chunk_length = P1M_chunks_idx_length[i][1]

            if chunk_length < 63:
                # adding monomer of type zero doesn't change the decimal number
                P1M_chunks_idx_length[i][1] += 1
            else:
                P1M_chunks[i].append(np.int64(0)) # type 0 is zero in decimals
                P1M_chunks_idx_length[i][1] = np.int64(1) # length 1
                P1M_chunks_idx_length[i][0] += 1 # new chunk idx






        elif rnd <= R[0]+R[1]: # P1 + M2 --> P2(+1) propagation rxn 5

            i = int(np.random.rand()*X[4])
            P1[i] += 1 # living polymer grows in 1 unit
            #P1M[i] = P1M[i]+'1' # M2 added

            P2.append(P1[i]) # The chain goes to P2
            P21.append(P11[i])
            #P2M.append(P1M[i])

            P1.pop(i)             # The P1 chain is deleted.
            P11.pop(i)            # The P11 chain is deleted.
            #P1M.pop(i)


            # Adding M2
            chunk_idx = P1M_chunks_idx_length[i][0]
            chunk_length = P1M_chunks_idx_length[i][1]

            if chunk_length < 63:
                P1M_chunks[i][chunk_idx] += 2 ** chunk_length
                P1M_chunks_idx_length[i][1] += 1
            else:
                P1M_chunks[i].append(np.int64(1)) # type 1 is one in decimals
                P1M_chunks_idx_length[i][1] = np.int64(1) # length 1
                P1M_chunks_idx_length[i][0] += 1 # new chunk idx

            popped_list, P1M_chunks = pop_nested_at_index(P1M_chunks,i)
            P2M_chunks.append(popped_list)

            popped_list, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,i)
            P2M_chunks_idx_length.append(popped_list)


            X[4] -= 1 # 1 P1 less
            X[5] += 1 # 1 P2 up
            X[3] -= 1                            # 1 M2 less





        elif rnd <= R[0]+R[1]+R[2]: #P2 + M1 --> P1(+1)  propagation rxn 6

            i = int(np.random.rand()*X[5])
            P2[i] += 1 # living polymer grows in 1 unit
            P21[i] += 1 # One M1 is adding to the chain.
            #P2M[i] = P2M[i]+'0'

            P1.append(P2[i])
            P11.append(P21[i])
            #P1M.append(P2M[i])

            P2.pop(i)
            P21.pop(i)
            #P2M.pop(i)


            # Adding M1
            chunk_length = P2M_chunks_idx_length[i][1]

            if chunk_length < 63:
                # adding monomer of type zero doesn't change the decimal number
                P2M_chunks_idx_length[i][1] += 1
            else:
                P2M_chunks[i].append(np.int64(0)) # type 0 is zero in decimals
                P2M_chunks_idx_length[i][1] = np.int64(1) # length 1
                P2M_chunks_idx_length[i][0] += 1 # new chunk idx

            popped_list, P2M_chunks = pop_nested_at_index(P2M_chunks,i)
            P1M_chunks.append(popped_list)

            popped_list, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,i)
            P1M_chunks_idx_length.append(popped_list)


            X[5] -= 1 # 1P2 -
            X[4] += 1 # 1P1 +
            X[2] -= 1 # 1M1 -






        elif rnd <= R[0]+R[1]+R[2]+R[3]: #P2 + M2 --> P2(+1) propagation rxn 7
            i = int(np.random.rand()*X[5])
            P2[i] += 1
            #P2M[i] = P2M[i]+'1'

            X[3] -= 1    # 1 M2 -


            # Adding M2
            chunk_idx = P2M_chunks_idx_length[i][0]
            chunk_length = P2M_chunks_idx_length[i][1]

            if chunk_length < 63:
                P2M_chunks[i][chunk_idx] += 2 ** chunk_length
                P2M_chunks_idx_length[i][1] += 1
            else:
                P2M_chunks[i].append(np.int64(1)) # type 1 is one in decimals
                P2M_chunks_idx_length[i][1] = np.int64(1) # length 1
                P2M_chunks_idx_length[i][0] += 1 # new chunk idx






        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]: #P1 + M1 --> P1(1) + L monomer transfer rxn 14
            i = int(np.random.rand()*X[4])
            L.append(P1[i])
            L1.append(P11[i])
            #LM.append(P1M[i])

            P1[i] = 1
            P11[i] = 1
            #P1M[i] = '0'


            # Adding P1 to dead chains
            LM_chunks.append(P1M_chunks[i])
            LM_chunks_idx_length.append(P1M_chunks_idx_length[i])

            # Changing the chain to P1(1)

            restarted_chain = List.empty_list(types.int64) # making sure we have the correct object type
            restarted_chain.append(np.int64(0))
            P1M_chunks[i] = restarted_chain # type 0 is zero in decimals
            P1M_chunks_idx_length[i][1] = np.int64(1) # length 1
            P1M_chunks_idx_length[i][0] = 0 # new chunk idx


            X[2] -=  1    # one monomer less
            X[6] +=  1    # one dead polymer formed






        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]: #P2 + M1 --> L + P1(1) monomer transfer rxn 15

            i = int(np.random.rand()*X[5])
            L.append(P2[i])
            L1.append(P21[i])
            #LM.append(P2M[i])

            P2[i] = 1
            P21[i] = 1
            #P2M[i] = '0'

            P1.append(P2[i])
            P11.append(P21[i])
            #P1M.append(P2M[i])

            P2.pop(i)
            P21.pop(i)
            #P2M.pop(i)


            # Popping chain from P2M and adding it to dead chains
            popped_list, P2M_chunks = pop_nested_at_index(P2M_chunks,i)
            LM_chunks.append(popped_list)
            popped_list, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,i)
            LM_chunks_idx_length.append(popped_list)

            # Adding new P1(1) to P1M
            new_list = List()
            new_list.append(np.int64(0)) # type 0 is zero in decimals
            P1M_chunks.append(new_list)

            new_list = List()
            new_list.append(np.int64(0)) # new chunk idx
            new_list.append(np.int64(1)) # length 1
            P1M_chunks_idx_length.append(new_list)


            X[2] -=  1    # one M1 less
            X[5] -=  1    # one P2 less
            X[6] +=  1    # one L formed
            X[4] +=  1    # one P1 more





        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]: #P2 + M2 --> L + P2(1) monomer transfer rxn 16

            i = int(np.random.rand()*X[5])
            L.append(P2[i])
            L1.append(P21[i])
            #LM.append(P2M[i])

            P2[i] = 1
            P21[i] = 0
            #P2M[i] = '1'


            # Adding P2 to dead chains
            LM_chunks.append(P2M_chunks[i])
            LM_chunks_idx_length.append(P2M_chunks_idx_length[i])

            # Changing the chain to P2(1)
            restarted_chain = List.empty_list(types.int64) # making sure we have the correct object type
            restarted_chain.append(np.int64(1))
            P2M_chunks[i] = restarted_chain # type 1 is one in decimals
            P2M_chunks_idx_length[i][1] = np.int64(1) # length 1
            P2M_chunks_idx_length[i][0] = 0 # new chunk idx

            X[3] -=  1    # one monomer less
            X[6] +=  1    # one dead polymer formed




        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]: #P1 + M2 --> L + P2(1) monomer transfer rxn 17

            i = int(np.random.rand()*X[4])
            L.append(P1[i])
            L1.append(P11[i])
            #LM.append(P1M[i])

            P1[i] = 1
            P11[i] = 0
            #P1M[i] = '1'

            P2.append(P1[i])
            P21.append(P11[i])
            #P2M.append(P1M[i])

            P1.pop(i)
            P11.pop(i)
            #P1M.pop(i)


            # Popping chain from P1M and adding it to dead chains
            popped_list, P1M_chunks = pop_nested_at_index(P1M_chunks,i)
            LM_chunks.append(popped_list)
            popped_list, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,i)
            LM_chunks_idx_length.append(popped_list)

            # Adding new P2(1) to P2M
            new_list = List()
            new_list.append(np.int64(1)) # type 1 is one in decimals
            P2M_chunks.append(new_list)

            new_list = List()
            new_list.append(np.int64(0)) # new chunk idx
            new_list.append(np.int64(1)) # length 1
            P2M_chunks_idx_length.append(new_list)



            X[3] -=  1   # 1M2 -
            X[5] +=  1   # 1P2 +
            X[6] +=  1   # 1L +
            X[4] -=  1   # 1P1 -







        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]: #R + M1 --> P1 initiation rxn 2
            P1.append(1)
            P11.append(1)
            #P1M.append('0') # M1 added


            # Adding new P1(1) to P1M
            new_list = List()
            new_list.append(np.int64(0)) # type 0 is zero in decimals
            P1M_chunks.append(new_list)

            new_list = List()
            new_list.append(np.int64(0)) # new chunk idx
            new_list.append(np.int64(1)) # length 1
            P1M_chunks_idx_length.append(new_list)



            X[4] += 1 # 1P1+
            X[1] -= 1 # 1R -
            X[2] -= 1 # 1M1-






        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]: #R + M2 --> P2 initiation rxn 3

            P2.append(1)
            P21.append(0)
            #P2M.append('1') # M2 added

            # Adding new P2(1) to P2M
            new_list = List()
            new_list.append(np.int64(1)) # type 1 is one in decimals
            P2M_chunks.append(new_list)

            new_list = List()
            new_list.append(np.int64(0)) # new chunk idx
            new_list.append(np.int64(1)) # length 1
            P2M_chunks_idx_length.append(new_list)

            X[5] += 1 # formation of P2
            X[1] -= 1 # consumption of 1 radical R
            X[3] -= 1 # consumption of 1 monomer M2





        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]+R[10]: # initiator decomposition rxn 1

            X[0] -= 1
            rn = np.random.rand()
            if rn <= f:
                X[1] += 2







        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]+R[10]+R[11]: # termination by combination rxn 8
            #P1 + P1 --> L(1+1)
            m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[4])
            if m == n:
                if m == 0:
                    n += 1
                else:
                    n -= 1
            L.append(P1[m] + P1[n])
            L1.append(P11[m]+P11[n]) # also saving the number of M1 in L
            #LM.append(P1M[m]+P1M[n])


            if m > n:
                ind = [m, n]
            else:
                ind = [n, m]
            for index in ind:
                P1.pop(index)
                P11.pop(index)
                #P1M.pop(index)


            # I reversed the chunk order before adding them together
            # Reversing of individual chunk order will happen when decoding to plot
            # Both positions and sizes of chunks at the junctions will be recorded,
            # so they don't go beyond 63 and we know when we have a junction point by combination

            # Popping chains from P1M, reversing the second, merging them, and adding the full chain to dead chains
            popped_list_1, P1M_chunks = pop_nested_at_index(P1M_chunks,ind[0])
            popped_list_2, P1M_chunks = pop_nested_at_index(P1M_chunks,ind[1])
            LM_chunks.append(extend_int64_list(popped_list_1,reverse_int64_list(popped_list_2)))

            # no need to reverse the index, we only need the index in the first chunk(junction), its length, and the length of the following chunk
            popped_list_1, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,ind[0])
            popped_list_2, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,ind[1])
            popped_list_1.append(popped_list_2[1])
            LM_chunks_idx_length.append(popped_list_1)



            X[4] -= 2 # 2P1 -
            X[6] += 1 # 1L +

        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]+R[10]+R[11]+R[12]: # termination by combination rxn 9
            #P1 + P2 --> L(1+2)
            m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[5])
            L.append(P1[m] + P2[n])
            L1.append(P11[m] + P21[n])
            #LM.append(P1M[m]+P2M[n])

            P1.pop(m)
            P11.pop(m)
            #P1M.pop(m)
            P2.pop(n)
            P21.pop(n)
            #P2M.pop(n)




            # I reversed the chunk order before adding them together
            # Reversing of individual chunk order will happen when decoding to plot
            # Both positions and sizes of chunks at the junctions will be recorded,
            # so they don't go beyond 63 and we know when we have a junction point by combination

            # Popping chains from P1M, reversing the second, merging them, and adding the full chain to dead chains
            popped_list_1, P1M_chunks = pop_nested_at_index(P1M_chunks,m)
            popped_list_2, P2M_chunks = pop_nested_at_index(P2M_chunks,n)
            LM_chunks.append(extend_int64_list(popped_list_1,reverse_int64_list(popped_list_2)))

            # no need to reverse the index, we only need the index in the first chunk(junction), its length, and the length of the following chunk
            popped_list_1, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,m)
            popped_list_2, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,n)
            popped_list_1.append(popped_list_2[1])
            LM_chunks_idx_length.append(popped_list_1)





            X[4] -= 1 # 1P1 -
            X[5] -= 1 # 1P2 -
            X[6] += 1 # formation of dead polymer

        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]+R[10]+R[11]+R[12]+R[13]: # rxn 10
            #P2 + P2 --> L(2+2)
            m, n = int(np.random.rand()*X[5]), int(np.random.rand()*X[5])
            if m == n:
                if m == 0:
                    n += 1
                else:
                    n -= 1

            X[5] -= 2 # 2P2 -
            X[6] += 1 # 1L +

            if m > n:
                ind = [m, n]
            else:
                ind = [n, m]
            L.append(P2[m] + P2[n])
            L1.append(P21[m] + P21[n])
            #LM.append(P2M[m]+P2M[n])

            for index in ind:
                P2.pop(index)
                P21.pop(index)
                #P2M.pop(index)


            # I reversed the chunk order before adding them together
            # Reversing of individual chunk order will happen when decoding to plot
            # Both positions and sizes of chunks at the junctions will be recorded,
            # so they don't go beyond 63 and we know when we have a junction point by combination

            # Popping chains from P1M, reversing the second, merging them, and adding the full chain to dead chains
            popped_list_1, P2M_chunks = pop_nested_at_index(P2M_chunks,ind[0])
            popped_list_2, P2M_chunks = pop_nested_at_index(P2M_chunks,ind[1])
            LM_chunks.append(extend_int64_list(popped_list_1,reverse_int64_list(popped_list_2)))

            # no need to reverse the index, we only need the index in the first chunk(junction), its length, and the length of the following chunk
            popped_list_1, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,ind[0])
            popped_list_2, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,ind[1])
            popped_list_1.append(popped_list_2[1])
            LM_chunks_idx_length.append(popped_list_1)







        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]+R[10]+R[11]+R[12]+R[13]+R[14]: # rxn 11
            # termination by disproportionation P1 + P1 --> L + L
            m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[4])
            if m == n:
                if m == 0:
                    n += 1
                else:
                    n -= 1

            L.append(P1[m])
            L1.append(P11[m])
            #LM.append(P1M[m])
            L.append(P1[n])
            L1.append(P11[n])
            #LM.append(P1M[n])


            if m > n:
                ind = [m, n]
            else:
                ind = [n, m]
            for index in ind:
                P1.pop(index)
                P11.pop(index)
                #P1M.pop(index)

                # Popping chain from P1M and adding it to dead chains
                popped_list, P1M_chunks = pop_nested_at_index(P1M_chunks,index)
                LM_chunks.append(popped_list)
                popped_list, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,index)
                LM_chunks_idx_length.append(popped_list)

            X[4] -= 2 # living polymer consumption
            X[6] += 2 # formation of dead polymer








        elif rnd <= R[0]+R[1]+R[2]+R[3]+R[4]+R[5]+R[6]+R[7]+R[8]+R[9]+R[10]+R[11]+R[12]+R[13]+R[14]+R[15]: # rxn 12
            # termination by disproportionation P1 + P2 --> L + L
            m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[5])
            L.append(P1[m])
            L1.append(P11[m])
            #LM.append(P1M[m])
            L.append(P2[n])
            L1.append(P21[n])
            #LM.append(P2M[n])

            P1.pop(m)
            P11.pop(m)
            #P1M.pop(m)
            P2.pop(n)
            P21.pop(n)
            #P2M.pop(n)

            # Popping chain from P1M and adding it to dead chains
            popped_list, P1M_chunks = pop_nested_at_index(P1M_chunks,m)
            LM_chunks.append(popped_list)
            popped_list, P1M_chunks_idx_length = pop_nested_at_index(P1M_chunks_idx_length,m)
            LM_chunks_idx_length.append(popped_list)


            # Popping chain from P2M and adding it to dead chains
            popped_list, P2M_chunks = pop_nested_at_index(P2M_chunks,n)
            LM_chunks.append(popped_list)
            popped_list, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,n)
            LM_chunks_idx_length.append(popped_list)

            X[4] -= 1   # 1P1 -
            X[5] -= 1   # 1P2 -
            X[6] += 2   # 2L +






        else: # termination by disproportionation P2 + P2 --> L + L rxn 13
            m, n = int(np.random.rand()*X[5]), int(np.random.rand()*X[5])
            if m == n:
                if m == 0:
                    n += 1
                else:
                    n -= 1

            L.append(P2[m])
            L1.append(P21[m])
            #LM.append(P2M[m])
            L.append(P2[n])
            L1.append(P21[n])
            #LM.append(P2M[n])

            if m > n:
                ind = [m, n]
            else:
                ind = [n, m]
            for index in ind:
                P2.pop(index)
                P21.pop(index)
                #P2M.pop(index)

                # Popping chain from P2M and adding it to dead chains
                popped_list, P2M_chunks = pop_nested_at_index(P2M_chunks,index)
                LM_chunks.append(popped_list)
                popped_list, P2M_chunks_idx_length = pop_nested_at_index(P2M_chunks_idx_length,index)
                LM_chunks_idx_length.append(popped_list)

            X[5] -= 2 # 2P2 -
            X[6] += 2 # 2L +

        t += dt
        T = -1.22354790e-09*t**3+7.06515946e-06*t**2-1.32627026e-02*t+3.11872289e+02
        #Tu = ((M1-X[2]) * dH1 / NA ) +  ( (M2-X[3]) * dH2 /NA)
        #Tl = (mM10 * Cp1/ MW1)  + (mM20 * Cp2 / MW2)  + (mW0 * Cpw / MWW)

        #T += Tu / Tl     # Updating temperature

        M1 , M2 = X[2],X[3]

        wm1 = (X[2]*MW1/NA)/(mW0+(X[2]*MW1/NA)+(X[3]*MW2/NA))    # updating wms
        wm2 = (X[3]*MW2/NA)/(mW0+(X[2]*MW1/NA)+(X[3]*MW2/NA))


    return L,L1,xt,x1,x2,wm1,wm2,t,LM_chunks,LM_chunks_idx_length


#Array with amount of molecules for each specie


X = np.zeros(n, dtype='int64')  # number of I, M, polymers, and ...
X[0] = int(CI0 * NA * V)        # Initiator    [I] = 0.001
X[1] = 0                        # Radical
X[2] = int(C10 * NA * V)       # Monomer1 (AM) [M1] = 4.0942
X[3] = int(C20 * NA * V)       # Monomer2 (AA)  [M2] = 1.7306
X[4] = 0                        # living polymer P1: polymers with monomer 1 as the chain end
X[5] = 0                        # living polymer P2
X[6] = 0                        # Dead polymer l

M10,M20 = X[2],X[3] # number of molecules of M1 and M2 at t = 0
W0 = CW*V*NA # molecules of water at t = 0
mW0 = CW*V*MWW # [g] mass of water at t = 0 and at control Volume
mM10 = C10*V*MW1
mM20 = C20*V*MW2
print(mW0,W0)
wm1 = ((X[2]*MW1)/NA)/(mW0+((X[2]*MW1)/NA)+((X[3]*MW2)/NA))
wm2 = ((X[3]*MW2)/NA)/(mW0+((X[2]*MW1)/NA)+((X[3]*MW2)/NA))
wm10 = wm1
wm20 = wm2

t = 0.0
R = np.zeros(nr, dtype=float)  # vector of reaction rates
xt = 0 # reaction conversion

# Initial and next memory tracking threshold - comment if using jit (incompatible function with numba)
init_real_time = time.time()
record_memory(0.0,t,mem_log)
next_track = 0.0 + tracking_step

a = datetime.datetime.now()
L,L1,xt,x1,x2,wm1,wm2,t,LM_chunks,LM_chunks_idx_length = MonteCarlo(t,f,X,R,M10,M20,tend,wm1,wm2,LM_chunks,P1M_chunks,P2M_chunks,LM_chunks_idx_length,P1M_chunks_idx_length,P2M_chunks_idx_length)
b = datetime.datetime.now()

# Final memory usage tracking - comment if using jit (incompatible function with numba)
record_memory(time.time()-init_real_time,tend,mem_log)

c = b-a
print(c)



# Microstructure

save_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
save_folder = save_path + save_time + note + '/'
os.makedirs(save_folder, exist_ok=True)


# Saving the decimal representation as a numpy object (.npy files tend to occupy less disk storage than text-based files like csv, and they load faster)

LM_chunks_np_obj = np.array([np.array(chain, dtype=np.int64) for chain in LM_chunks], dtype=object)
LM_chunks_idx_length_np_obj = np.array([np.array(chain, dtype=np.int64) for chain in LM_chunks_idx_length], dtype=object)

np.save(f'{save_folder}decimal_chains_file_vartemp_alpha1_vol_5_10-16.npy', LM_chunks_np_obj)
np.save(f'{save_folder}idx_length_file_vartemp_alpha1_vol_5_10-16.npy', LM_chunks_idx_length_np_obj)

# Saving memory usage tracking
print(mem_log)
mem_log.to_csv(save_folder+"memory_profile.csv", index=False)

'''
# testing loading file
loaded_obj = np.load(f'{save_folder}idx_length_file_vartemp_alpha1_vol_5_10-16.npy', allow_pickle=True)
print(LM_chunks_idx_length_np_obj)
print(loaded_obj)
'''
