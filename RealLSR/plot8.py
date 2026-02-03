import os
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams["font.size"] = 15
if os.path.isdir("./Plots/plot8")==False: os.makedirs("./Plots/plot8")

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
color = ["r","g","b","c","gold","m"]
qset  = [0.125,0.25,0.5,1,2]

def func(DATAs,METHs):
    for DATA in DATAs:
        res = np.zeros((6,len(SIZEs),SEED))
        # ERM
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",DATA,SIZEs[si],0.0,se), delimiter=",")
            res[0,si,se] = np.log(tmp[np.argmin(tmp[:,-2]),-1])
        # Bridge
        for me in range(5):
            for si in range(len(SIZEs)):
                tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-2])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[me+1,si,se] = tmp2[se]
        # plot
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        #
        k = 0
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label="ERM", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 1
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.125$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 2
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.25$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 3
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.5$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 4
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=1$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 5
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=2$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        ax.legend(loc='upper right', fontsize=20); plt.xticks(range(len(SIZEs)),SIZEs); plt.tight_layout()
        plt.xlabel(r'training data size', fontsize=20)
        plt.ylabel(r'log(test risk)', fontsize=20)
        plt.savefig("./Plots/plot8/%s_loss.png"%DATA, bbox_inches="tight", pad_inches=.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
func(DATAs,METHs)
#
METHs = ["NBridge1-0.1250000000/relu-4-5","NBridge1-0.2500000000/relu-4-5","NBridge1-0.5000000000/relu-4-5","NBridge1-1.0000000000/relu-4-5","NBridge1-2.0000000000/relu-4-5",
         "NBridge1-0.1250000000/relu-4-20","NBridge1-0.2500000000/relu-4-20","NBridge1-0.5000000000/relu-4-20","NBridge1-1.0000000000/relu-4-20","NBridge1-2.0000000000/relu-4-20"]
func(DATAs,METHs)