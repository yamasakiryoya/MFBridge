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
color = ["red","green","blue","cyan","magenta","gold","black"]#["magenta","gold","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [100]

def func(TYPEs,METHs,METHs2):
    for me,TYPE,si in itertools.product(range(len(METHs)),TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); ay = ax.twinx()
        #
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-2])
        optla = LAMs2[np.argmin(np.mean(res[la,:]))]
        LC = np.zeros((2000,4,20))
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,SIZEs[si],optla,se), delimiter=",")
            LC[:,0,se] = np.log(tmp[:,0])
            LC[:,1,se] = np.log(tmp[:,-2])
            LC[:,2,se] = tmp[:,1]
        #
        ax.plot(range(1,2001), np.mean(LC[:,0,:], axis=1), lw=1.5, color=color[0], linestyle='dotted')
        ax.fill_between(range(1,2001), np.quantile(LC[:,0,:], axis=1, q=0.25), np.quantile(LC[:,0,:], axis=1, q=0.75), color=color[0], alpha=0.3, edgecolor=None)
        ax.plot(range(1,2001), np.mean(LC[:,1,:], axis=1), lw=1.5, color=color[1], linestyle='dashed')
        ax.fill_between(range(1,2001), np.quantile(LC[:,1,:], axis=1, q=0.25), np.quantile(LC[:,1,:], axis=1, q=0.75), color=color[1], alpha=0.3, edgecolor=None)
        ay.plot(range(1,2001), np.mean(LC[:,2,:], axis=1), lw=1.5, color=color[2], linestyle='dashdot')
        ay.fill_between(range(1,2001), np.quantile(LC[:,2,:], axis=1, q=0.25), np.quantile(LC[:,2,:], axis=1, q=0.75), color=color[2], alpha=0.3, edgecolor=None)
        #
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        ax.plot(1+np.ones(10)*np.argmin(np.mean(LC[:,1,:], axis=1)), np.linspace(BOT,TOP,10), color="k", lw=1.5, linestyle="--", zorder=3)
        #
        ax.set_ylim(BOT,TOP); ax.set_xlabel('epoch'); ax.set_ylabel(r'log(MSE)'); ay.set_ylabel(r'regularizer'); plt.tight_layout()
        plt.savefig("./Plots/plot5/%s_%s_Rloss.png"%(METHs2[me],TYPE), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
METHs2= ["NGLasso1-0.0000000000"   ,"NGLasso1-0.2500000000"   ,"NGLasso1-0.5000000000"   ,"NGLasso1-0.7500000000"   ,"NGLasso1-1.0000000000",
         "NBridge1-0.1250000000-10","NBridge1-0.2500000000-10","NBridge1-0.5000000000-10","NBridge1-1.0000000000-10","NBridge1-2.0000000000-10"]
func(TYPEs,METHs,METHs2)
