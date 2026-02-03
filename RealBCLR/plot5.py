import os
import itertools
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
if os.path.isdir("./Plots/plot5")==False: os.makedirs("./Plots/plot5")
color = ["red","green","blue","cyan","magenta","gold","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [100]
#==================================================#
def func(TYPEs,METHs,METHs2):
    for me,TYPE,si in itertools.product(range(len(METHs)),TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); ay = ax.twinx()
        #
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        optla = LAMs2[np.argmin(np.mean(res[la,:]))]
        LC = np.zeros((500,4,20))
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],optla,se), delimiter=",")
            LC[:,0,se] = tmp[:,0]
            LC[:,1,se] = tmp[:,-4]
            LC[:,2,se] = tmp[:,1]
        #
        ax.plot(range(1,501), np.mean(LC[:,0,:], axis=1), lw=1.5, color=color[0], linestyle='dotted')
        ax.fill_between(range(1,501), np.quantile(LC[:,0,:], axis=1, q=0.25), np.quantile(LC[:,0,:], axis=1, q=0.75), color=color[0], alpha=0.3, edgecolor=None)
        ax.plot(range(1,501), np.mean(LC[:,1,:], axis=1), lw=1.5, color=color[1], linestyle='dashed')
        ax.fill_between(range(1,501), np.quantile(LC[:,1,:], axis=1, q=0.25), np.quantile(LC[:,1,:], axis=1, q=0.75), color=color[1], alpha=0.3, edgecolor=None)
        ay.plot(range(1,501), np.mean(LC[:,2,:], axis=1), lw=1.5, color=color[2], linestyle='dashdot')
        ay.fill_between(range(1,501), np.quantile(LC[:,2,:], axis=1, q=0.25), np.quantile(LC[:,2,:], axis=1, q=0.75), color=color[2], alpha=0.3, edgecolor=None)
        #
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        ax.plot(1+np.ones(10)*np.argmin(np.mean(LC[:,1,:], axis=1)), np.linspace(BOT,TOP,10), color="k", lw=1.5, linestyle="--", zorder=3)
        #
        ax.set_ylim(BOT,TOP); ax.set_xlabel('epoch'); ax.set_ylabel(r'MCEE'); ay.set_ylabel(r'regularizer'); plt.tight_layout()
        plt.savefig("./Plots/plot5/%s_%s_Closs.png"%(METHs2[me],TYPE), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
METHs2= ["NGLasso1-0.0000000000"   ,"NGLasso1-0.2500000000"   ,"NGLasso1-0.5000000000"   ,"NGLasso1-0.7500000000"   ,"NGLasso1-1.0000000000",
         "NBridge1-0.1250000000-10","NBridge1-0.2500000000-10","NBridge1-0.5000000000-10","NBridge1-1.0000000000-10","NBridge1-2.0000000000-10"]
func(TYPEs,METHs,METHs2)
#==================================================#
def func(TYPEs,METHs,METHs2):
    for me,TYPE,si in itertools.product(range(len(METHs)),TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); ay = ax.twinx()
        #
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-3])
        optla = LAMs2[np.argmin(np.mean(res[la,:]))]
        LC = np.zeros((500,4,20))
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],optla,se), delimiter=",")
            LC[:,0,se] = tmp[:,3]
            LC[:,1,se] = tmp[:,-3]
            LC[:,2,se] = tmp[:,1]
        #
        ax.plot(range(1,501), np.mean(LC[:,0,:], axis=1), lw=1.5, color=color[0], linestyle='dotted')
        ax.fill_between(range(1,501), np.quantile(LC[:,0,:], axis=1, q=0.25), np.quantile(LC[:,0,:], axis=1, q=0.75), color=color[0], alpha=0.3, edgecolor=None)
        ax.plot(range(1,501), np.mean(LC[:,1,:], axis=1), lw=1.5, color=color[1], linestyle='dashed')
        ax.fill_between(range(1,501), np.quantile(LC[:,1,:], axis=1, q=0.25), np.quantile(LC[:,1,:], axis=1, q=0.75), color=color[1], alpha=0.3, edgecolor=None)
        ay.plot(range(1,501), np.mean(LC[:,2,:], axis=1), lw=1.5, color=color[2], linestyle='dashdot')
        ay.fill_between(range(1,501), np.quantile(LC[:,2,:], axis=1, q=0.25), np.quantile(LC[:,2,:], axis=1, q=0.75), color=color[2], alpha=0.3, edgecolor=None)
        #
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        ax.plot(1+np.ones(10)*np.argmin(np.mean(LC[:,1,:], axis=1)), np.linspace(BOT,TOP,10), color="k", lw=1.5, linestyle="--", zorder=3)
        #
        ax.set_ylim(BOT,TOP); ax.set_xlabel('epoch'); ax.set_ylabel(r'MCR'); ay.set_ylabel(r'regularizer'); plt.tight_layout()
        plt.savefig("./Plots/plot5/%s_%s_miscla.png"%(METHs2[me],TYPE), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
METHs2= ["NGLasso1-0.0000000000"   ,"NGLasso1-0.2500000000"   ,"NGLasso1-0.5000000000"   ,"NGLasso1-0.7500000000"   ,"NGLasso1-1.0000000000",
         "NBridge1-0.1250000000-10","NBridge1-0.2500000000-10","NBridge1-0.5000000000-10","NBridge1-1.0000000000-10","NBridge1-2.0000000000-10"]
func(TYPEs,METHs,METHs2)
