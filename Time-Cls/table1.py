import numpy as np
import itertools

for CUDA in [0]:
    print("1ERM",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1ERM/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1MFBridge",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1MFBridge/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1MFElim",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1MFElim/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1MFLoss",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1MFLoss/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1MFRisk",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1MFRisk/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1MFGrad",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1MFGrad/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1SPINN",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1SPINN/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1STG",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1STG/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1LassoNet",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1LassoNet/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1Bridge",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1Bridge/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")
    print("1SpAM",end="&")
    for DIM,M in itertools.product([10,20,40],[10,20,40]):
        arr = np.zeros(100)
        for SEED in range(100): arr[SEED] = np.loadtxt(f"Results/1SpAM/{SEED}-{DIM}-{M}-{CUDA}.csv")
        print("$%.2f\\pm%.2f$"%(np.mean(arr),np.std(arr)),end="&")
    print("\\\\")