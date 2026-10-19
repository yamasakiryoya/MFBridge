import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess

for SEED in range(100):
    for DIM in [10,20,40]:
        for M in [10,20,40]:
            for CUDA in [0]:
                print("1Bridge",SEED,M,CUDA)
                subprocess.run(f"python 1Bridge.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1SpAM",SEED,M,CUDA)
                subprocess.run(f"python 1SpAM.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1ERM",SEED,M,CUDA)
                subprocess.run(f"python 1ERM.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1MFBridge",SEED,M,CUDA)
                subprocess.run(f"python 1MFBridge.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1MFLoss",SEED,M,CUDA)
                subprocess.run(f"python 1MFLoss.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1MFRisk",SEED,M,CUDA)
                subprocess.run(f"python 1MFRisk.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1MFElim",SEED,M,CUDA)
                subprocess.run(f"python 1MFElim.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1MFGrad",SEED,M,CUDA)
                subprocess.run(f"python 1MFGrad.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1SPINN",SEED,M,CUDA)
                subprocess.run(f"python 1SPINN.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1LassoNet",SEED,M,CUDA)
                subprocess.run(f"python 1LassoNet.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)
                print("1STG",SEED,M,CUDA)
                subprocess.run(f"python 1STG.py --type quasi-multicolinear1 --dim {DIM} --size 50 --alp 0.5 --lam 1 --Mnum {M} --act relu --lay 4 --seed {SEED} --cuda {CUDA}", shell=True)