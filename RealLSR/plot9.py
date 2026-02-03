import os
import itertools
import numpy as np
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
if os.path.isdir("./Plots/plot9")==False: os.makedirs("./Plots/plot9")
tasks = ["P"]
colors = ["red","green","blue","cyan","magenta","gold"]
datas = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZE  = 100

def func1(datas):
    for t in range(len(tasks)):
        task = tasks[t]
        allres = np.zeros((len(datas),6,SEED))
        for d in range(len(datas)):
            data = datas[d]
            # ERM
            for se in range(SEED):
                tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(data,SIZE,0.0,se), delimiter=",")
                allres[d,0,se] = tmp[np.argmin(tmp[:,-2]),-1]
            # MF-Bridge
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
            for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,1,se] = tmp2[se]
            # NN-SGLasso
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4"]
            for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,2,se] = tmp2[se]
            # Lasso
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["LBridge1-1.0000000000"]
            for me,la,se in itertools.product(range(1),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,3,se] = tmp2[se]
            # SpAM
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["ABridge1-1.0000000000/relu-4"]
            for me,la,se in itertools.product(range(1),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,4,se] = tmp2[se]
            # RandomForest1
            for se in range(SEED):
                tmp = np.loadtxt("./Results/RandomForest1/%s/size-%d/1-seed-%d.csv"%(data,SIZE,se), delimiter=",")
                allres[d,5,se] = tmp[-1]


        import scikit_posthocs as sp
        import pandas as pd

        fig = plt.figure(figsize=(9,2.5)); ax = fig.add_subplot(111)#1.7
        algorithms = ["ERM", "MFBridge", "NNSGLasso", "Lasso", "SpAM", "random forest"]
        colors = ["red","green","blue","cyan","magenta","gold"]

        D2, M2, T2 = allres.shape
        rows = []
        for d2 in range(D2):
            for t2 in range(T2):
                for m2, alg in enumerate(algorithms):
                    rows.append({
                        'dataset': d2,
                        'method': alg,
                        'trial': t2,
                        'score': float(allres[d2, m2, t2])
                    })

        df = pd.DataFrame(rows)
        df['dataset'] = df['dataset'].astype(int)
        df['trial'] = df['trial'].astype(int)
        df['block_id'] = df['dataset'] * 1000 + df['trial']
        #
        test_results = sp.posthoc_conover_friedman(df,block_col='dataset',block_id_col='block_id',group_col='method',y_col='score',melted=True)
        #
        df['rank'] = df.groupby(['dataset','trial'])['score'].rank(ascending=True, method='average')
        avg_rank = df.groupby('method')['rank'].mean().loc[algorithms]
        #
        left_to_right_order = avg_rank.sort_values().index.tolist()
        color_dict = dict(zip(algorithms, colors))
        ordered_colors = [color_dict[alg] for alg in left_to_right_order]
        sp.critical_difference_diagram(avg_rank.loc[left_to_right_order],test_results,alpha=0.05,label_fmt_left='{label} [{rank:.3f}]  ',label_fmt_right='  [{rank:.3f}] {label}',color_palette=ordered_colors,ax=ax)
        #
        plt.title(r"MSE")
        plt.tight_layout()
        plt.savefig("./Plots/plot9/cdd-Rloss-%s.png"%task, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

func1(datas)