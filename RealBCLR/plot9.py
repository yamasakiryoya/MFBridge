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
datas = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]

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
                allres[d,0,se] = tmp[np.argmin(tmp[:,-4]),-2]
            # MF-Bridge
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
            for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,1,se] = tmp2[se]
            # NN-SGLasso
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4"]
            for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,2,se] = tmp2[se]
            # Lasso
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["LBridge1-1.0000000000"]
            for me,la,se in itertools.product(range(1),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,3,se] = tmp2[se]
            # SpAM
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["ABridge1-1.0000000000/relu-4"]
            for me,la,se in itertools.product(range(1),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,4,se] = tmp2[se]
            # RandomForest1
            for se in range(SEED):
                tmp = np.loadtxt("./Results/RandomForest1/%s/size-%d/1-seed-%d.csv"%(data,SIZE,se), delimiter=",")
                allres[d,5,se] = tmp[-2]


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
        plt.title(r"MCEE")
        plt.tight_layout()
        plt.savefig("./Plots/plot9/cdd-Closs-%s.png"%task, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

func1(datas)

def func2(datas):
    for t in range(len(tasks)):
        task = tasks[t]
        allres = np.zeros((len(datas),6,SEED))
        for d in range(len(datas)):
            data = datas[d]
            # ERM
            for se in range(SEED):
                tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(data,SIZE,0.0,se), delimiter=",")
                allres[d,0,se] = tmp[np.argmin(tmp[:,-3]),-1]
            # MF-Bridge
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
            for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,1,se] = tmp2[se]
            # NN-SGLasso
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["NGLasso1-0.0000000000/relu-4","NGLasso1-0.2500000000/relu-4","NGLasso1-0.5000000000/relu-4","NGLasso1-0.7500000000/relu-4","NGLasso1-1.0000000000/relu-4"]
            for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,2,se] = tmp2[se]
            # Lasso
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["LBridge1-1.0000000000"]
            for me,la,se in itertools.product(range(1),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; allres[d,3,se] = tmp2[se]
            # SpAM
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            METHs = ["ABridge1-1.0000000000/relu-4"]
            for me,la,se in itertools.product(range(1),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],data,SIZE,LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
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
        plt.title(r"MCR")
        plt.tight_layout()
        plt.savefig("./Plots/plot9/cdd-miscla-%s.png"%task, bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

func2(datas)
