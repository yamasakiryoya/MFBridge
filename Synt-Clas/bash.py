import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import itertools
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def MP(run_script,tasks):
    if rank == 0:
        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        for i in range(1, size):
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=i, tag=1)
                task_index += 1
            else:
                comm.send(None, dest=i, tag=0)
                closed_workers += 1
        while closed_workers < num_workers:
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            if task_index < len(tasks):
                comm.send(tasks[task_index], dest=source, tag=1)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=0)
                closed_workers += 1
    else:
        while True:
            status = MPI.Status()
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == 0: break
            run_script(task)
            comm.send(True, dest=0, tag=2)


def run(params):
    code = params[0]
    if code==0:
        _, TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFBridge.py --type %s --dim %d --size %d --alp 1.0 --lam 0. --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==1:
        _, TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFRisk.py --type %s --dim %d --size %d --alp 1.0 --lam 0. --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==2:
        _, TYPE,DIM,SIZE,SEED,CUDA = params
        subprocess.run("python 1RandomForest.py --type %s --dim %d --size %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==3:
        _, TYPE,DIM,SIZE,ALP,LAM,SEED,CUDA = params
        subprocess.run("python 1Bridge.py --type %s --dim %d --size %d --alp %f --lam %s --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==4:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1SpAM.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==5:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFBridge.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==6:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFLoss.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==7:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFRisk.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==8:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFElim.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==9:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFGrad.py --type %s --dim %d --size %d --alp %f --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==10:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1SPINN.py --type %s --dim %d --size %d --alp %f --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==11:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1LassoNet.py --type %s --dim %d --size %d --alp %f --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==12:
        _, TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1STG.py --type %s --dim %d --size %d --alp %f --lam %s --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==13:
        _, TYPE,DIM,SIZE,SCORE,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2Filter.py --type %s --dim %d --size %d --score %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,SCORE,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==14:
        _, TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2MFBridge.py --type %s --dim %d --size %d --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==15:
        _, TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2MFBridge0.py --type %s --dim %d --size %d --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==16:
        _, TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2MFRisk0.py --type %s --dim %d --size %d --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==17:
        _, TYPE,DIM,SIZE,SEED,CUDA = params
        subprocess.run("python 1Const.py --type %s --dim %d --size %d --seed %d --cuda %d"%(TYPE,DIM,SIZE,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)


DATA = "quasi-multicolinear1"
ACT  = "relu"
LAY  = 4
CUDA = -1
#
tasks = []
tasks.extend(list(itertools.product([ 0],[DATA],[ 3],[25,50,100,200,400],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 0],[DATA],[10],[25,50,100,200,400],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 1],[DATA],[10],[25,50,100,200,400],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 2],[DATA],[10],[25,50,100,200,400],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 3],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 4],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 5],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 6],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 7],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 8],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([ 9],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([10],[DATA],[10],[25,50,100,200,400],[0.0,0.25,0.5,0.75,1.0], [10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([11],[DATA],[10],[25,50,100,200,400],[10.**-1,10**-.5,10.**0,10**.5,10.**1],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([12],[DATA],[10],[25,50,100,200,400],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
MP(run, tasks)
#
tasks = []
tasks.extend(list(itertools.product([13],[DATA],[10],[25,50,100,200,400],[0,1,2],[0.],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([13],[DATA],[10],[25,50,100,200,400],[3,4],[10.**(i/8.-6.) for i in range(81)],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([14],[DATA],[10],[25,50,100,200,400],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([15],[DATA],[10],[25,50,100,200,400],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([16],[DATA],[10],[25,50,100,200,400],[ACT],[LAY],range(100),[CUDA])))
tasks.extend(list(itertools.product([17],[DATA],[10],[25,50,100,200,400],range(100),[CUDA])))
MP(run, tasks)
#
def run_cmd(cmd):
    subprocess.run(cmd, shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
#
tasks = []
for J in range(10): tasks.append("python summary1.py --data %s --j %d" % (DATA, J))
tasks.append("python summary2.py --data %s" % DATA)
MP(run_cmd, tasks)
#
tasks = []
for name in ["2step","classic","main","MFBridge","output","previous"]:
    tasks.append("python plot1-%s.py --data %s" % (name, DATA))
for SIZE in [25,50,100,200,400]:
    for name in ["2step","classic","main","MFBridge","output","previous"]:
        tasks.append("python plot4-%s.py --data %s --size %d" % (name, DATA, SIZE))
    for J in [0]:
        tasks.append("python plot2.py --data %s --size %d --j %d" % (DATA, SIZE, J))
        tasks.append("python plot3.py --data %s --size %d --j %d" % (DATA, SIZE, J))
        tasks.append("python plot5.py --data %s --size %d --j %d" % (DATA, SIZE, J))
MP(run_cmd, tasks)