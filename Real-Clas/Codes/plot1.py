import os
import itertools
import numpy as np
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import scikit_posthocs as sp
import pandas as pd
from scipy import stats
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 15
plt.rcParams['text.usetex'] = True
if os.path.isdir("./Plots/plot1")==False: os.makedirs("./Plots/plot1")
datas = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","dis","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]

SEED  = 10
SIZE  = 100

def rank_heatmap_of_mean_score(allres, datas, algor, savepath, figsize=None, title=""):
    """
    Rows = methods (algor), columns = datasets (datas).
    Unlike rank_heatmap (which ranks within each trial, then averages the
    rank over trials), this first averages the score over trials for each
    (dataset, method), and then ranks methods on that averaged score. This
    is the aggregation order used for critical_difference_diagram-style
    comparisons, where each dataset contributes a single representative
    value per method.
    allres: array of shape (D2, M2, T2) = (datasets, methods, trials)
    """
    D2, M2, T2 = allres.shape
    assert M2 == len(algor)
    assert D2 == len(datas)

    #average the score over trials first -> one value per (dataset, method)
    mean_score = allres.mean(axis=2)  # (D2, M2)

    #then rank methods within each dataset on that averaged score
    #(ascending: smaller score = better = rank 1)
    grid_ds_method = np.apply_along_axis(
        lambda x: st.rankdata(x, method='average'), axis=1, arr=mean_score
    )  # (D2, M2)

    #transpose to (methods, datasets) for the requested orientation
    grid = grid_ds_method.T  # (M2, D2)

    if figsize is None:
        figsize = (max(6, 0.35*D2), max(4, 0.35*M2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=M2)

    for i in range(M2):
        for j in range(D2):
            val = grid[i, j]
            norm = (val - 1) / max(M2 - 1, 1)
            txt_color = 'white' if (norm > 0.65 or norm < 0.2) else 'black'
            ax.text(j, i, "%d"%val, ha='center', va='center', fontsize=6, color=txt_color)

    ax.set_xticks(range(D2)); ax.set_xticklabels(datas, rotation=90, fontsize=6)
    ax.set_yticks(range(M2)); ax.set_yticklabels(algor, fontsize=8)
    ax.set_xlabel("Dataset"); ax.set_ylabel("Method")
    if title: ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Rank of mean error (1 = best)")

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", pad_inches=.02, dpi=200,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def rank_heatmap(allres, datas, algor, savepath, figsize=None, title=""):
    """
    Rows = methods (algor), columns = datasets (datas).
    Each cell shows the average rank of that method on that dataset
    (averaged over the trial axis of allres), colored by a colormap.
    allres: array of shape (D2, M2, T2) = (datasets, methods, trials)
    Ranking convention matches the rest of this script: for each
    (dataset, trial), methods are ranked 1..M2 by score (ascending =
    smaller is better), ties given the average rank; the rank is then
    averaged over trials to give one number per (dataset, method).
    """
    D2, M2, T2 = allres.shape
    assert M2 == len(algor)
    assert D2 == len(datas)

    #rank over the METHOD axis for each (dataset, trial), matching
    #df.groupby(['dataset','trial'])['score'].rank(...) used elsewhere in this script
    arr = allres.transpose(0, 2, 1)  # (D2, T2, M2)
    ranks = np.apply_along_axis(lambda x: st.rankdata(x, method='average'), axis=2, arr=arr)
    mean_rank = ranks.mean(axis=1)  # average over trials -> (D2, M2)

    #transpose to (methods, datasets) for the requested orientation
    grid = mean_rank.T  # (M2, D2)

    if figsize is None:
        figsize = (max(6, 0.35*D2), max(4, 0.35*M2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=M2)

    #write the rank value inside each cell, switching text color for readability
    for i in range(M2):
        for j in range(D2):
            val = grid[i, j]
            norm = (val - 1) / max(M2 - 1, 1)
            txt_color = 'white' if (norm > 0.65 or norm < 0.2) else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=6, color=txt_color)

    ax.set_xticks(range(D2)); ax.set_xticklabels(datas, rotation=90, fontsize=6)
    ax.set_yticks(range(M2)); ax.set_yticklabels(algor, fontsize=8)
    ax.set_xlabel("Dataset"); ax.set_ylabel("Method")
    if title: ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Average rank (1 = best)")

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", pad_inches=.02, dpi=200,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def func1(datas):
    allres = np.zeros((len(datas),8,SEED))
    for d in range(len(datas)):
        data = datas[d]
        allres[d, 0] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("ERM",data),delimiter=",")
        allres[d, 1] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFBridge",data),delimiter=",")
        allres[d, 2] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFElim",data),delimiter=",")
        allres[d, 3] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1SPINN",data),delimiter=",")
        allres[d, 4] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1Bridge",data),delimiter=",")
        allres[d, 5] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2MFBridge0",data),delimiter=",")
        allres[d, 6] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2MFBridge",data),delimiter=",")
        allres[d, 7] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2Filter-hscic",data),delimiter=",")
        
    fig = plt.figure(figsize=(4.5,3)); ax = fig.add_subplot(111)#1.7
    algor = ["ERM","MFBridge","MFElim","SPINN","Bridge","ERM2","MFBridge2","HSCIC2"]
    color = ["red","green","blue","cyan","magenta","gold","purple","darkorange"]

    D2, M2, T2 = allres.shape
    rows = []
    for d2 in range(D2):
        for t2 in range(T2):
            for m2, alg in enumerate(algor):
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
    avg_rank = df.groupby('method')['rank'].mean().loc[algor]
    #
    left_to_right_order = avg_rank.sort_values().index.tolist()
    color_dict = dict(zip(algor, color))
    ordered_color = [color_dict[alg] for alg in left_to_right_order]
    sp.critical_difference_diagram(avg_rank.loc[left_to_right_order],test_results,alpha=0.05,label_fmt_left='{label} [{rank:.3f}]  ',label_fmt_right='  [{rank:.3f}] {label}',color_palette=ordered_color,ax=ax)
    #
    plt.title(r"MCE")
    plt.tight_layout()
    plt.savefig("./Plots/plot1/cdd-CMCE-main.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
    #
    rank_heatmap(allres, datas, algor, "./Plots/plot1/rankheatmap-CMCE-main.png", title=r"MCE")
    rank_heatmap_of_mean_score(allres, datas, algor, "./Plots/plot1/rankheatmap-meanscore-CMCE-main.png", title=r"MCE")
    #
    return allres

res1 = func1(datas)

def func2(datas):
    allres = np.zeros((len(datas),8,SEED))
    for d in range(len(datas)):
        data = datas[d]
        allres[d, 0] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("ERM",data),delimiter=",")
        allres[d, 1] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFBridge",data),delimiter=",")
        allres[d, 2] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFElim",data),delimiter=",")
        allres[d, 3] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1SPINN",data),delimiter=",")
        allres[d, 4] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1Bridge",data),delimiter=",")
        allres[d, 5] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2MFBridge0",data),delimiter=",")
        allres[d, 6] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2MFBridge",data),delimiter=",")
        allres[d, 7] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2Filter-hscic",data),delimiter=",")
        
    fig = plt.figure(figsize=(4.5,3)); ax = fig.add_subplot(111)#1.7
    algor = ["ERM","MFBridge","MFElim","SPINN","Bridge","ERM2","MFBridge2","HSCIC2"]
    color = ["red","green","blue","cyan","magenta","gold","purple","darkorange"]

    D2, M2, T2 = allres.shape
    rows = []
    for d2 in range(D2):
        for t2 in range(T2):
            for m2, alg in enumerate(algor):
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
    avg_rank = df.groupby('method')['rank'].mean().loc[algor]
    #
    left_to_right_order = avg_rank.sort_values().index.tolist()
    color_dict = dict(zip(algor, color))
    ordered_color = [color_dict[alg] for alg in left_to_right_order]
    sp.critical_difference_diagram(avg_rank.loc[left_to_right_order],test_results,alpha=0.05,label_fmt_left='{label} [{rank:.3f}]  ',label_fmt_right='  [{rank:.3f}] {label}',color_palette=ordered_color,ax=ax)
    #
    plt.title(r"MCR")
    plt.tight_layout()
    plt.savefig("./Plots/plot1/cdd-CMCR-main.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
    #
    rank_heatmap(allres, datas, algor, "./Plots/plot1/rankheatmap-CMCR-main.png", title=r"MCR")
    rank_heatmap_of_mean_score(allres, datas, algor, "./Plots/plot1/rankheatmap-meanscore-CMCR-main.png", title=r"MCR")
    #
    return allres

res2 = func2(datas)

def func3(datas):
    allres = np.zeros((len(datas),25,SEED))
    for d in range(len(datas)):
        data = datas[d]
        allres[d, 0] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("ERM",data),delimiter=",")
        allres[d, 1] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFBridge",data),delimiter=",")
        allres[d, 2] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCE.csv"%("1MFBridge",0.125,data),delimiter=",")
        allres[d, 3] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCE.csv"%("1MFBridge",0.25,data),delimiter=",")
        allres[d, 4] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCE.csv"%("1MFBridge",0.5,data),delimiter=",")
        allres[d, 5] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCE.csv"%("1MFBridge",1,data),delimiter=",")
        allres[d, 6] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCE.csv"%("1MFBridge",2,data),delimiter=",")
        allres[d, 7] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFElim",data),delimiter=",")
        allres[d, 8] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFLoss",data),delimiter=",")
        allres[d, 9] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFRisk",data),delimiter=",")
        allres[d,10] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1MFGrad",data),delimiter=",")
        allres[d,11] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1SPINN",data),delimiter=",")
        allres[d,12] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1STG",data),delimiter=",")
        allres[d,13] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1LassoNet",data),delimiter=",")
        allres[d,14] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1Bridge",data),delimiter=",")
        allres[d,15] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1SpAM",data),delimiter=",")
        allres[d,16] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("1RandomForest",data),delimiter=",")
        allres[d,17] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2MFBridge0",data),delimiter=",")
        allres[d,18] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2MFBridge",data),delimiter=",")
        allres[d,19] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2MFRisk0",data),delimiter=",")
        allres[d,20] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2Filter-f_classif",data),delimiter=",")
        allres[d,21] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2Filter-mutual_info_classif",data),delimiter=",")
        allres[d,22] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2Filter-hsic",data),delimiter=",")
        allres[d,23] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2Filter-chsic",data),delimiter=",")
        allres[d,24] = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("2Filter-hscic",data),delimiter=",")
        


    fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111)#1.7
    algor = ["ERM","MFBridge",r"MFBridge ($q\!=\!0.125$)",r"MFBridge ($q\!=\!0.25$)",r"MFBridge ($q\!=\!0.5$)",r"MFBridge ($q\!=\!1$)",r"MFBridge ($q\!=\!2$)","MFElim","MFLoss","MFRisk","MFGrad","SPINN","STG","LassoNet","Bridge","SpAM","Random Forest","ERM2","MFBridge2","PI2","ANOVA-F2","MI2","HSIC2","COND2","HSCIC2"]
    color = ["red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown","red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown","red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown"]

    D2, M2, T2 = allres.shape
    rows = []
    for d2 in range(D2):
        for t2 in range(T2):
            for m2, alg in enumerate(algor):
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
    avg_rank = df.groupby('method')['rank'].mean().loc[algor]
    #
    left_to_right_order = avg_rank.sort_values().index.tolist()
    color_dict = dict(zip(algor, color))
    ordered_color = [color_dict[alg] for alg in left_to_right_order]
    sp.critical_difference_diagram(avg_rank.loc[left_to_right_order],test_results,alpha=0.05,label_fmt_left='{label} [{rank:.3f}]  ',label_fmt_right='  [{rank:.3f}] {label}',color_palette=ordered_color,ax=ax)
    #
    plt.title(r"MCE")
    plt.tight_layout()
    plt.savefig("./Plots/plot1/cdd-CMCE-full.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
    #
    rank_heatmap(allres, datas, algor, "./Plots/plot1/rankheatmap-CMCE-full.png", title=r"MCE")
    rank_heatmap_of_mean_score(allres, datas, algor, "./Plots/plot1/rankheatmap-meanscore-CMCE-full.png", title=r"MCE")
    #
    return allres

res3 = func3(datas)

def func4(datas):
    allres = np.zeros((len(datas),25,SEED))
    for d in range(len(datas)):
        data = datas[d]
        allres[d, 0] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("ERM",data),delimiter=",")
        allres[d, 1] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFBridge",data),delimiter=",")
        allres[d, 2] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCR.csv"%("1MFBridge",0.125,data),delimiter=",")
        allres[d, 3] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCR.csv"%("1MFBridge",0.25,data),delimiter=",")
        allres[d, 4] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCR.csv"%("1MFBridge",0.5,data),delimiter=",")
        allres[d, 5] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCR.csv"%("1MFBridge",1,data),delimiter=",")
        allres[d, 6] = np.loadtxt("Results/summary/%s-%.10f-%s-CMCR.csv"%("1MFBridge",2,data),delimiter=",")
        allres[d, 7] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFElim",data),delimiter=",")
        allres[d, 8] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFLoss",data),delimiter=",")
        allres[d, 9] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFRisk",data),delimiter=",")
        allres[d,10] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1MFGrad",data),delimiter=",")
        allres[d,11] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1SPINN",data),delimiter=",")
        allres[d,12] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1STG",data),delimiter=",")
        allres[d,13] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1LassoNet",data),delimiter=",")
        allres[d,14] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1Bridge",data),delimiter=",")
        allres[d,15] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1SpAM",data),delimiter=",")
        allres[d,16] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("1RandomForest",data),delimiter=",")
        allres[d,17] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2MFBridge0",data),delimiter=",")
        allres[d,18] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2MFBridge",data),delimiter=",")
        allres[d,19] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2MFRisk0",data),delimiter=",")
        allres[d,20] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2Filter-f_classif",data),delimiter=",")
        allres[d,21] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2Filter-mutual_info_classif",data),delimiter=",")
        allres[d,22] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2Filter-hsic",data),delimiter=",")
        allres[d,23] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2Filter-chsic",data),delimiter=",")
        allres[d,24] = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("2Filter-hscic",data),delimiter=",")
        
    fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111)#1.7
    algor = ["ERM","MFBridge",r"MFBridge ($q\!=\!0.125$)",r"MFBridge ($q\!=\!0.25$)",r"MFBridge ($q\!=\!0.5$)",r"MFBridge ($q\!=\!1$)",r"MFBridge ($q\!=\!2$)","MFElim","MFLoss","MFRisk","MFGrad","SPINN","STG","LassoNet","Bridge","SpAM","Random Forest","ERM2","MFBridge2","PI2","ANOVA-F2","MI2","HSIC2","COND2","HSCIC2"]
    color = ["red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown","red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown","red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown"]

    D2, M2, T2 = allres.shape
    rows = []
    for d2 in range(D2):
        for t2 in range(T2):
            for m2, alg in enumerate(algor):
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
    avg_rank = df.groupby('method')['rank'].mean().loc[algor]
    #
    left_to_right_order = avg_rank.sort_values().index.tolist()
    color_dict = dict(zip(algor, color))
    ordered_color = [color_dict[alg] for alg in left_to_right_order]
    sp.critical_difference_diagram(avg_rank.loc[left_to_right_order],test_results,alpha=0.05,label_fmt_left='{label} [{rank:.3f}]  ',label_fmt_right='  [{rank:.3f}] {label}',color_palette=ordered_color,ax=ax)
    #
    plt.title(r"MCR")
    plt.tight_layout()
    plt.savefig("./Plots/plot1/cdd-CMCR-full.png", bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
    #
    rank_heatmap(allres, datas, algor, "./Plots/plot1/rankheatmap-CMCR-full.png", title=r"MCR")
    rank_heatmap_of_mean_score(allres, datas, algor, "./Plots/plot1/rankheatmap-meanscore-CMCR-full.png", title=r"MCR")
    #
    return allres

res4 = func4(datas)

