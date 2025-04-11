import os
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams["font.size"] = 15
if os.path.isdir("./Plots/plot6")==False: os.makedirs("./Plots/plot6")

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
#"""
def func(DATAs,METHs):
    for DATA in DATAs:
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[0],DATA,SIZEs[si],0.0,se), delimiter=",")
            res[si,se] = np.log(tmp[np.argmin(tmp[:,-2]),-1])
        #
        bp0 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="r", linewidth=1.5), whiskerprops=dict(color="r", linewidth=1.5), capprops=dict(color="r", linewidth=1.5), flierprops=dict(markeredgecolor="r", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp0['boxes']: b.set_edgecolor("r"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label="ERM", color="r")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="r", marker="o", zorder=3)
        # GSLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-2])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp1 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="g", linewidth=1.5), whiskerprops=dict(color="g", linewidth=1.5), capprops=dict(color="g", linewidth=1.5), flierprops=dict(markeredgecolor="g", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp1['boxes']: b.set_edgecolor("g"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label="NNSGLasso", color="g")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="g", marker="o", zorder=3)
        # Bridge
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me+5],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-2])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me+5],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="b", linewidth=1.5), whiskerprops=dict(color="b", linewidth=1.5), capprops=dict(color="b", linewidth=1.5), flierprops=dict(markeredgecolor="b", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("b"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label="MFBridge", color="b")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="b", marker="o", zorder=3)
        #
        ax.legend(loc='upper right', fontsize=20); plt.xticks(range(len(SIZEs)),SIZEs); plt.tight_layout()
        plt.xlabel(r'training data size', fontsize=20)
        plt.ylabel(r'log(test risk)', fontsize=20)
        plt.savefig("./Plots/plot6/%s_loss.png"%DATA, bbox_inches="tight", pad_inches=.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
func(DATAs,METHs)