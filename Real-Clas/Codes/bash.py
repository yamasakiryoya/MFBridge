#python bash.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import multiprocessing
from multiprocessing import Pool
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
    #lambda=0
    if code==0:
        _,DATA,SIZE,SEED,CUDA = params
        subprocess.run("python 1RandomForest.py --data %s --size %d --seed %d --cuda %d"%(DATA,SIZE,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==1:
        _,DATA,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFBridge.py --data %s --size %d --alp 1.0 --lam 0. --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==2:
        _,DATA,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFRisk.py --data %s --size %d --alp 1.0 --lam 0. --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    #regularized
    if code==3:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1Bridge.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==4:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1SpAM.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    #alternative
    if code==5:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFBridge.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==6:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFLoss.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==7:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFRisk.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==8:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFElim.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==9:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1MFGrad.py --data %s --size %d --alp %s --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    #previous
    if code==10:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1SPINN.py --data %s --size %d --alp %s --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==11:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1LassoNet.py --data %s --size %d --alp %s --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==12:
        _,DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 1STG.py --data %s --size %d --alp %s --lam %s --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ALP,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    #2-step
    if code==13:
        _,DATA,SIZE,SCORE,LAM,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2Filter.py --data %s --size %d --score %d --lam %s --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,SCORE,LAM,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==14:
        _,DATA,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2MFBridge.py --data %s --size %d --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==15:
        _,DATA,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2MFBridge0.py --data %s --size %d --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
    if code==16:
        _,DATA,SIZE,ACT,LAY,SEED,CUDA = params
        subprocess.run("python 2MFRisk0.py --data %s --size %d --Mnum 20 --act %s --lay %d --seed %d --cuda %d"%(DATA,SIZE,ACT,LAY,SEED,CUDA), shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)


DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]

tasks = []
for DATA in DATAs:
    tasks.extend(list(itertools.product([ 0],[DATA],[100],range(10),[-1])))
    tasks.extend(list(itertools.product([ 1],[DATA],[100],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 2],[DATA],[100],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 3],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 4],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    #
    tasks.extend(list(itertools.product([ 5],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 6],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 7],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 8],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([ 9],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    #
    tasks.extend(list(itertools.product([10],[DATA],[100],[0.0,0.25,0.50,0.75,1.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([11],[DATA],[100],[10.**-1,10.**-.5,10.**0,10.**.5,10.**1],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([12],[DATA],[100],[0.125,0.25,0.5,1.0,2.0],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([13],[DATA],[100],[0,1,2],[0.],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([13],[DATA],[100],[3,4],[10.**(i/8.-6.) for i in range(81)],["relu"],[4],range(10),[-1])))
    #
    tasks.extend(list(itertools.product([14],[DATA],[100],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([15],[DATA],[100],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([16],[DATA],[100],["relu"],[4],range(10),[-1])))
MP(run, tasks)
#
tasks = []
for DATA in DATAs:
    tasks.extend(list(itertools.product([14],[DATA],[100],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([15],[DATA],[100],["relu"],[4],range(10),[-1])))
    tasks.extend(list(itertools.product([16],[DATA],[100],["relu"],[4],range(10),[-1])))
MP(run, tasks)
#
def run_cmd(cmd):
    subprocess.run(cmd, shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
tasks = []
for DATA in DATAs:
    for J in range(10): tasks.append("python summary1.py --data %s --j %d" % (DATA, J))
    tasks.append("python summary2.py --data %s" % DATA)
MP(run_cmd, tasks)
#
if rank == 0: subprocess.run("python plot1.py", shell=True, stdout=open("worker_%d.log" % rank, "a"), stderr=subprocess.STDOUT)
