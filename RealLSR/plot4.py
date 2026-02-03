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
color = ["red","green","blue","cyan","magenta","gold","black"]#["magenta","gold","orange","saddlebrown","gray"]#

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
            fs[0,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[0,:,se] /= np.max(fs[0,:,se])
        # MFBridge
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-2])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[1,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[1,:,se] /= np.max(fs[1,:,se])
        # GSLasso
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-2])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[2,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[2,:,se] /= np.max(fs[2,:,se])
        # Lasso
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-2])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[3,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[3,:,se] /= np.max(fs[3,:,se])
        # SpAM
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-2])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[4,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[4,:,se] /= np.max(fs[4,:,se])
        # random forest
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest1/%s/size-%d/1-seed-%d.csv"%(TYPE,SIZEs[si],se), delimiter=",")
            fs[5,:,se] = tmp[1:-2]; fs[5,:,se] /= np.max(fs[5,:,se])
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
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n=%d$'%SIZEs[si]); plt.tight_layout()#test risk#
        plt.savefig("./Plots/plot4/%s_Rloss.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
func(TYPEs)
