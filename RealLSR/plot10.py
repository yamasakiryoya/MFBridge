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
if os.path.isdir("./Plots/plot10")==False: os.makedirs("./Plots/plot10")
tasks = ["P"]
colors = ["red","green","blue","cyan","magenta","gold"]
datas = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZE  = 100

for ALP in [0.9,0.95,0.99]:

    def func1(datas):
        for t in range(len(tasks)):
            task = tasks[t]
            allres = np.zeros((len(datas),10,SEED))
            for d in range(len(datas)):
                data = datas[d]
                # ERM
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/NBridge3-%.10f/relu-4-10/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,0,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # MF-Bridge
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/NBridge2-%.10f/relu-4-10/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,1,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # NN-SGLasso
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/NGLasso2-%.10f/relu-4/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,2,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # Lasso
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/LBridge2-%.10f/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,3,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # Lasso
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/LBridge3-%.10f/relu-4-10/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,4,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # SpAM
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/ABridge2-%.10f/relu-4/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,5,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # SpAM
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/ABridge3-%.10f/relu-4-10/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,6,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # RandomForest1
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/RandomForest2-%.10f/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,7,se] = tmp[-1]
                # RandomForest1
                for se in range(SEED):
                    tmp = np.loadtxt("./Results/RandomForest3-%.10f/relu-4-10/%s/size-%d/1-seed-%d.csv"%(ALP,data,SIZE,se), delimiter=",")
                    allres[d,8,se] = tmp[np.argmin(tmp[:,-2]),-1]
                # MF-Bridge
                tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
                METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
                for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), tmp[np.argmin(tmp[:,-2]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,9,se] = tmp2[se]


            import scikit_posthocs as sp
            import pandas as pd

            fig = plt.figure(figsize=(9,3.5)); ax = fig.add_subplot(111)#1.7
            algorithms = ["FS-ERM(NN)","FS-MFBridge(NN)","FS-NNSGLasso(NN)","FS-Lasso('')","FS-Lasso(NN)","FS-SpAM('')","FS-SpAM(NN)","FS-random forest('')","FS-random forest(NN)","MFBridge"]
            cm = plt.get_cmap('jet')
            colors = [cm(k/9) for k in range(10)]#["red","green","blue","cyan","magenta","gold","brown","orange","violet","k"]

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
            plt.savefig("./Plots/plot10/cdd-Rloss-%s-%s.png"%(ALP,task), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

    func1(datas)