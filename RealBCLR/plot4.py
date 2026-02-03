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
if os.path.isdir("./Plots/plot4")==False: os.makedirs("./Plots/plot4")
color = ["red","green","blue","cyan","magenta","gold","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [100]
#
def func(TYPEs):
    for TYPE,si in itertools.product(TYPEs,range(len(SIZEs))):
        XYful = np.genfromtxt("../Datasets/%s.csv"%TYPE, delimiter=",", skip_header=1)
        DIM = len(XYful[0,:])-1
        #
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        fs = np.zeros((6,DIM,SEED))
        # ERM
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,SIZEs[si],0.,se), delimiter=",")
            fs[0,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[0,:,se] /= np.max(fs[0,:,se])
        # MFBridge
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[1,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[1,:,se] /= np.max(fs[1,:,se])
        # GSLasso
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[2,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[2,:,se] /= np.max(fs[2,:,se])
        # Lasso
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[3,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[3,:,se] /= np.max(fs[3,:,se])
        # SpAM
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[4,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[4,:,se] /= np.max(fs[4,:,se])
        # random forest
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest1/%s/size-%d/1-seed-%d.csv"%(TYPE,SIZEs[si],se), delimiter=",")
            fs[5,:,se] = tmp[2:-4]; fs[5,:,se] /= np.max(fs[5,:,se])
        #
        k = 0
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"ERM", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"MFBridge", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"NNSGLasso", facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 3
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"Lasso", facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 4
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"SpAM", facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 5
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot4/%s_Closs.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        fs = np.zeros((6,DIM,SEED))
        # ERM
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,SIZEs[si],0.,se), delimiter=",")
            fs[0,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[0,:,se] /= np.max(fs[0,:,se])
        # MFBridge
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[1,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[1,:,se] /= np.max(fs[1,:,se])
        # GSLasso
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[2,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[2,:,se] /= np.max(fs[2,:,se])
        # Lasso
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[3,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[3,:,se] /= np.max(fs[3,:,se])
        # SpAM
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[4,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[4,:,se] /= np.max(fs[4,:,se])
        # random forest
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest1/%s/size-%d/1-seed-%d.csv"%(TYPE,SIZEs[si],se), delimiter=",")
            fs[5,:,se] = tmp[2:-4]; fs[5,:,se] /= np.max(fs[5,:,se])
        #
        k = 0
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"ERM", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"MFBridge", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"NNSGLasso", facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 3
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"Lasso", facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 4
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"SpAM", facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 5
        for l in range(DIM):
            ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1.5, color=color[k])
            plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1.5, color=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot4/%s_miscla.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
func(TYPEs)
