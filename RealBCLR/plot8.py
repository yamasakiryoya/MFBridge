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
SIZEs = [100]
color = ["r","g","b","c","gold","m"]
qset  = [0.125,0.25,0.5,1,2]
#
def func(DATAs,METHs):
    for DATA in DATAs:
        #=====#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # ERM
        res = np.zeros((6,len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",DATA,SIZEs[si],0.0,se), delimiter=",")
            res[0,si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        """
        # Bridge
        for me in range(5):
            for si in range(len(SIZEs)):
                tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-4])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[me+1,si,se] = tmp2[se]
        #
        """
        k = 0
        bp0 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp0['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label="ERM", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        """
        k = 1
        bp1 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.125$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 2
        bp2 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.25$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 3
        bp3 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.5$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 4
        bp4 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=1$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 5
        bp5 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=2$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        """
        ax.legend(loc='upper right', fontsize=20); plt.xticks(range(len(SIZEs)),SIZEs); plt.tight_layout()
        plt.xlabel(r'training data size', fontsize=20); plt.ylabel('test risk', fontsize=20)
        plt.savefig("./Plots/plot8/%s_loss.png"%DATA, bbox_inches="tight", pad_inches=.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #=====#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # ERM
        res = np.zeros((6,len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",DATA,SIZEs[si],0.0,se), delimiter=",")
            res[0,si,se] = tmp[np.argmin(tmp[:,-3]),-1]
        """
        # Bridge
        for me in range(5):
            for si in range(len(SIZEs)):
                tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-3])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[me+1,si,se] = tmp2[se]
        #
        """
        k = 0
        bp0 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp0['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label="ERM", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        """
        k = 1
        bp1 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.125$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 2
        bp2 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.25$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 3
        bp3 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=0.5$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 4
        bp4 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=1$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        k = 5
        bp5 = ax.boxplot([res[k,0]], positions=[0], patch_artist=True, widths=0.4, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res[k,:,:], axis=1), lw=1.5, label=r"$q=2$", color=color[k])
        ax.scatter(range(len(SIZEs)), np.mean(res[k,:,:], axis=1),s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        """
        ax.legend(loc='upper right', fontsize=20); plt.xticks(range(len(SIZEs)),SIZEs); plt.tight_layout()
        plt.xlabel(r'training data size', fontsize=20); plt.ylabel('test misclassification rate', fontsize=20)
        plt.savefig("./Plots/plot8/%s_miscla.png"%DATA, bbox_inches="tight", pad_inches=.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma's_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
func(DATAs,METHs)
#
METHs = ["NBridge1-0.1250000000/relu-4-5" ,"NBridge1-0.2500000000/relu-4-5" ,"NBridge1-0.5000000000/relu-4-5" ,"NBridge1-1.0000000000/relu-4-5" ,"NBridge1-2.0000000000/relu-4-5" ,
         "NBridge1-0.1250000000/relu-4-20","NBridge1-0.2500000000/relu-4-20","NBridge1-0.5000000000/relu-4-20","NBridge1-1.0000000000/relu-4-20","NBridge1-2.0000000000/relu-4-20"]
func(DATAs,METHs)
