# Extracted unchanged from original notebook cell 8.
import numpy as np
from scipy import interpolate

def FRwithFCC(alfcc):
# def FRwithFCC(alfcc,T,NS):
    # This function is an updated metric for freezing range, 
    # using the temperature at which the FCC aluminum
    # phase first develops 
    # rather than a fixed cutoff of 0.02 fraction solid. 
    # The index of this temperature is stored.
    # 
    # assume alfcc is monotonically increasing
    # 
    assert alfcc[0]<=alfcc[-1], "FCC phase is decreasing."
    # 
    t=[a > 1e-30 for a in alfcc]
    ind=[i for i, x in enumerate(t) if x]
    return ind[0]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def FR_cutoff(NS, T, cutoff):
    T_idx = find_nearest(NS, cutoff)
    T_cutoff = T[T_idx]
    FR_cutoff = round(T_cutoff - min(T), 5)

    return FR_cutoff

def CSC_3(NS, NH):
    NS = np.asarray(NS)
    NH = np.asarray(NH)
    t = []

    for i, e in enumerate(NH):
        t.append((e / NH[-1]) ** 2)
    # print(t)

    df1 = (np.abs(NS - 0.4)).argmin()
    df2 = (np.abs(NS - 0.9)).argmin()
    df3 = (np.abs(NS - 0.98)).argmin()
    # df4 = (np.abs(NS - 0.99)).argmin()
    # if (t[df4] > t[df3]):
    #     t3 = t[df4]
    # else:
    #     t3 = t[df3]


    t1 = t[df1]
    t2 = t[df2]
    t3 = t[df3]

    t_v = t3 - t2
    t_r = t2 - t1
    if (t_r==0):
        if (t_v==0):
            CSC=0
        else:
            CSC=1
    else:
        CSC = round((t_v / t_r), 5)

    return CSC

def interpolate_hist_data(
    NS_1, 
    NH_1, 
    dx=0.001,
):
    # use NS, solid phase fractuion as the precision rule
    x_0 = np.amin(NS_1)
    x_1 = np.amax(NS_1)
    x_list = np.arange(x_0, x_1, dx)
    if x_list[-1]<x_1:
        x_list = np.append(x_list, x_1)
    else:
        x_list[-1]=x_1
        
    f=interpolate.interp1d(NS_1, NH_1)
    y_list = f(x_list)
    return x_list, y_list

def CSC_4(NS_0, NH_0, dx=0.001):
    '''
    1. Here, assume NS and NH are calculated from one call
    and NS start from 0.0 and increase towards 1.0
    2. Assumption of translating heat into time: dQ/dt = k (1/sqrt(t))
    3. Note: NH is suspected to be the tot heat in the system. i.e., 
    when NS = 0.0 (tot liq), NH is at max. When NS = 1.0, NH is at min.
    for a Scheil calc.
    4. The grid points of NH is refined w.r.t the solid fractions
    to ensure it can calculate time interval well 
    '''
    assert NS_0[0]<=NS_0[-1], "Not intended: All-solid phase, NS, is decreasing!"
    assert NH_0[0]>=NH_0[-1], "Not intended: NH is increasing!"

    NS_1 = np.asarray(NS_0)
    NH_1 = np.asarray(NH_0)
    # make them dense
    NS, NH = interpolate_hist_data(
        NS_1, NH_1, dx,
    )
    
    # convert into fractional time: t in (0,1)
    t = []
    for i, this_Q in enumerate(NH):
        t.append(
            np.power( (NH[0]-this_Q)/(NH[0]-NH[-1]), 2 )
        )

    # apply the criterion
    df1 = (np.abs(NS - 0.4)).argmin()
    df2 = (np.abs(NS - 0.9)).argmin()
    df3 = (np.abs(NS - 0.98)).argmin()
    # df4 = (np.abs(NS - 0.99)).argmin()
    # if (t[df4] > t[df3]):
    #     t3 = t[df4]
    # else:
    #     t3 = t[df3]

    t1 = t[df1]
    t2 = t[df2]
    t3 = t[df3]

    t_v = t3-t2 # vulnerable time: NS 0.90 -> 0.98
    t_r = t2-t1 # relaxing time:   NS 0.40 -> 0.90
    
    # failure safe
    if t_r==0:
        # pass: not sure the idea here
        # maybe just use negative values as marks
        if t_v==0:
            CSC = 0 # eutatic case 
        else:
            CSC = 1 # CSC takes 1 as a quite large value for Al alloys
    else:
        CSC = round(t_v/t_r, 5)
    return CSC, t, t_v, t_r, t1, t2, t3
