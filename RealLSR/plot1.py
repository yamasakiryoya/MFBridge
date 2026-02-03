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
if os.path.isdir("./Plots/plot1")==False: os.makedirs("./Plots/plot1")
colors = ["red","green","blue","cyan","magenta","gold","black"]

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [100]

def func(TYPEs,METHs):
    for TYPE in TYPEs:
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],0.0,se), delimiter=",")
            res[si,se] = np.log(tmp[np.argmin(tmp[:,-2]),-1])
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=colors[0])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=colors[0])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=colors[0])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=colors[0])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[0])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="ERM", lw=1.5, facecolor='None', edgecolors=colors[0], zorder=3)
        # Bridge
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=colors[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=colors[1])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=colors[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=colors[1])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[1])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFBridge", lw=1.5, facecolor='None', edgecolors=colors[1], zorder=3)
        # GSLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me+5],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=colors[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=colors[2])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=colors[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=colors[2])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[2])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="^", label="NNSGLasso", lw=1.5, facecolor='None', edgecolors=colors[2], zorder=3)
        # Lasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            result = np.zeros((len(LAMs2),SEED))
            for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                tmp = np.loadtxt("./Results/LBridge1-1.0000000000/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=colors[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=colors[3])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=colors[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=colors[3])

        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[3])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="v", label="Lasso", lw=1.5, facecolor='None', edgecolors=colors[3], zorder=3)
        # SpAM
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            result = np.zeros((len(LAMs2),SEED))
            for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                tmp = np.loadtxt("./Results/ABridge1-1.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=colors[4])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=colors[4])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=colors[4])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=colors[4])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[4])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="<", label="SpAM", lw=1.5, facecolor='None', edgecolors=colors[4], zorder=3)
        # random forest
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/RandomForest1/%s/size-%d/1-seed-%d.csv"%(TYPE,SIZEs[si],se), delimiter=",")
            res[si,se] = np.log(tmp[-1])
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=colors[5])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=colors[5])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=colors[5])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=colors[5])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[5])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker=">", label="random forest", lw=1.5, facecolor='None', edgecolors=colors[5], zorder=3)
        # # NSCAD
        # res = np.zeros((len(SIZEs),SEED))
        # for si in range(len(SIZEs)):
        #     tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
        #     result = np.zeros((len(LAMs2),SEED))
        #     for se,la in itertools.product(range(SEED),range(len(LAMs2))):
        #         tmp = np.loadtxt("./Results/NSCAD1-3.0000000000/relu-4-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
        #         tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
        #         if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        # #
        # ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[1], linestyle="-.")
        # ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="NSCAD", lw=1.5, facecolor='None', edgecolors=colors[1], zorder=3)
        # # NMCP
        # res = np.zeros((len(SIZEs),SEED))
        # for si in range(len(SIZEs)):
        #     tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
        #     result = np.zeros((len(LAMs2),SEED))
        #     for se,la in itertools.product(range(SEED),range(len(LAMs2))):
        #         tmp = np.loadtxt("./Results/NMCP1-3.0000000000/relu-4-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
        #         tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
        #         if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        # #
        # ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[2], linestyle="-.")
        # ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="NMCP", lw=1.5, facecolor='None', edgecolors=colors[2], zorder=3)
        # # NLogSum
        # res = np.zeros((len(SIZEs),SEED))
        # for si in range(len(SIZEs)):
        #     tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
        #     result = np.zeros((len(LAMs2),SEED))
        #     for se,la in itertools.product(range(SEED),range(len(LAMs2))):
        #         tmp = np.loadtxt("./Results/NLogSum1-0.0000100000/relu-4-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
        #         tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
        #         if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        # #
        # ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=colors[3], linestyle="-.")
        # ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="NLogSum", lw=1.5, facecolor='None', edgecolors=colors[3], zorder=3)
        #
        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel(r'$n$'); plt.ylabel(r'log(MSE)'); plt.tight_layout()
        plt.savefig("./Plots/plot1/%s_Rloss.png"%TYPE, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"   ]
func(TYPEs,METHs)
