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
if os.path.isdir("./Plots/plot2")==False: os.makedirs("./Plots/plot2")
color = ["red","green","blue","cyan","magenta","gold","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [100]
#==================================================#
def func(TYPEs,METHs):
    for TYPE,si in itertools.product(TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-4]),-2]
        #
        k = 0
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.125$", lw=1.5, facecolor='None', edgecolor=color[k], marker="o", zorder=3)
        #
        k = 1
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.25$", lw=1.5, facecolor='None', edgecolor=color[k], marker="^", zorder=3)
        #
        k = 2
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.5$", lw=1.5, facecolor='None', edgecolor=color[k], marker="<", zorder=3)
        #
        k = 3
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=1$", lw=1.5, facecolor='None', edgecolor=color[k], marker="<", zorder=3)
        #
        k = 4
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=2$", lw=1.5, facecolor='None', edgecolor=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[2],TYPE,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([4*i for i in range(9)],[-5,-4,-3,-2,-1,0,1,2,3]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCEE, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot2/MFBridge_%s_Closs.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-3]),-1]
        #
        k = 0
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.125$", lw=1.5, facecolor='None', edgecolor=color[k], marker="o", zorder=3)
        #
        k = 1
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.25$", lw=1.5, facecolor='None', edgecolor=color[k], marker="^", zorder=3)
        #
        k = 2
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.5$", lw=1.5, facecolor='None', edgecolor=color[k], marker="v", zorder=3)
        #
        k = 3
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=1$", lw=1.5, facecolor='None', edgecolor=color[k], marker="<", zorder=3)
        #
        k = 4
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=2$", lw=1.5, facecolor='None', edgecolor=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[2],TYPE,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([4*i for i in range(9)],[-5,-4,-3,-2,-1,0,1,2,3]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCR, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot2/MFBridge_%s_miscla.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
func(TYPEs,METHs)
#==================================================#
def func(TYPEs,METHs):
    for TYPE,si in itertools.product(TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-4]),-2]
        #
        k = 0
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0$", lw=1.5, facecolor='None', edgecolor=color[k], marker="o", zorder=3)
        #
        k = 1
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.25$", lw=1.5, facecolor='None', edgecolor=color[k], marker="^", zorder=3)
        #
        k = 2
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.5$", lw=1.5, facecolor='None', edgecolor=color[k], marker="v", zorder=3)
        #
        k = 3
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.75$", lw=1.5, facecolor='None', edgecolor=color[k], marker="<", zorder=3)
        #
        k = 4
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=1$", lw=1.5, facecolor='None', edgecolor=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([4*i for i in range(9)],[-5,-4,-3,-2,-1,0,1,2,3]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCEE, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot2/NNSGLasso_%s_Closs.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-3]),-1]
        #
        k = 0
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0$", lw=1.5, facecolor='None', edgecolor=color[k], marker="o", zorder=3)
        #
        k = 1
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.25$", lw=1.5, facecolor='None', edgecolor=color[k], marker="^", zorder=3)
        #
        k = 2
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.5$", lw=1.5, facecolor='None', edgecolor=color[k], marker="v", zorder=3)
        #
        k = 3
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.75$", lw=1.5, facecolor='None', edgecolor=color[k], marker="<", zorder=3)
        #
        k = 4
        for l in range(len(LAMs2)):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(res[k,l], q=0.25), np.quantile(res[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res[k,l], q=0.75), lw=1.5, color=color[k])
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=1$", lw=1.5, facecolor='None', edgecolor=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([4*i for i in range(9)],[-5,-4,-3,-2,-1,0,1,2,3]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCR, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot2/NNSGLasso_%s_miscla.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"]
func(TYPEs,METHs)