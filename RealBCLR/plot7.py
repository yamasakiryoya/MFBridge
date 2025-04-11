import os
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams["font.size"] = 15
if os.path.isdir("./Plots/plot7")==False: os.makedirs("./Plots/plot7")

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
#
def func(DATAs,METHs):
    for DATA in DATAs:
        #=====#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # Bridge(M=5)
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s-5/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-4])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s-5/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="r", linewidth=1.5), whiskerprops=dict(color="r", linewidth=1.5), capprops=dict(color="r", linewidth=1.5), flierprops=dict(markeredgecolor="r", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("r"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label=r"MFBridge($M=5$)", color="r")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="r", marker="o", zorder=3)
        # Bridge(M=10)
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-4])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="g", linewidth=1.5), whiskerprops=dict(color="g", linewidth=1.5), capprops=dict(color="g", linewidth=1.5), flierprops=dict(markeredgecolor="g", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("g"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label=r"MFBridge($M=10$)", color="g")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="g", marker="o", zorder=3)
        # Bridge(M=20)
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s-20/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-4])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s-20/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="b", linewidth=1.5), whiskerprops=dict(color="b", linewidth=1.5), capprops=dict(color="b", linewidth=1.5), flierprops=dict(markeredgecolor="b", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("b"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label=r"MFBridge($M=20$)", color="b")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="b", marker="o", zorder=3)
        #
        ax.legend(loc='upper right', fontsize=20); plt.xticks(range(len(SIZEs)),SIZEs); plt.tight_layout()
        plt.xlabel(r'training data size', fontsize=20); plt.ylabel('test risk', fontsize=20)
        plt.savefig("./Plots/plot7/%s_loss.png"%(DATA,DIM), bbox_inches="tight", pad_inches=.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #=====#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # Bridge(M=5)
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s-5/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-3])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s-5/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="r", linewidth=1.5), whiskerprops=dict(color="r", linewidth=1.5), capprops=dict(color="r", linewidth=1.5), flierprops=dict(markeredgecolor="r", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("r"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label=r"MFBridge($M=5$)", color="r")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="r", marker="o", zorder=3)
        # Bridge(M=10)
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-3])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s-10/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="g", linewidth=1.5), whiskerprops=dict(color="g", linewidth=1.5), capprops=dict(color="g", linewidth=1.5), flierprops=dict(markeredgecolor="g", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("g"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label=r"MFBridge($M=10$)", color="g")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="g", marker="o", zorder=3)
        # Bridge(M=20)
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(20),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s-20/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs2[la],se), delimiter=",")
                    result[la,se] = np.min(tmp[:,-3])
                #
                LAMs3 = [10.**(i/16.+LAMs1[np.argmin(np.mean(result, axis=1))]) for i in [-3,-2,-1,0,1,2,3]]
                for se,la in itertools.product(range(SEED),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s-20/%s/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],DATA,SIZEs[si],LAMs3[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.4, medianprops=dict(color="b", linewidth=1.5), whiskerprops=dict(color="b", linewidth=1.5), capprops=dict(color="b", linewidth=1.5), flierprops=dict(markeredgecolor="b", markeredgewidth=1, marker="x"), boxprops={'fill': None})
        for b in bp2['boxes']: b.set_edgecolor("b"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)#
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, label=r"MFBridge($M=20$)", color="b")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1),s=60, lw=1.5, facecolor='None', edgecolors="b", marker="o", zorder=3)
        #
        ax.legend(loc='upper right', fontsize=20); plt.xticks(range(len(SIZEs)),SIZEs); plt.tight_layout()
        plt.xlabel(r'training data size', fontsize=20)#訓練データ数
        plt.ylabel('test misclassification rate', fontsize=20)#テスト誤分類率
        plt.savefig("./Plots/plot7/%s_miscla.png"%(DATA,DIM), bbox_inches="tight", pad_inches=.02, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
DATAs = ["banana","haberman","blood_transfusion_service_center","banknote_authentication","bupa_liver_disorders","irish","mammographic_masses","wilt","phoneme","monk3","monk1","monk2","penguins","raisin","auction_verification_result","rice_cammeo_osmancik","titanic","NHANES_age_prediction","HTRU_2","saheart","threeOf9","profb","breast_cancer_wisconsin_original","tic_tac_toe","xd6","indian_liver_patient","parity5+5","mofn_3_7_10","magic_gamma_telescope","Customer_Churn","credit_approval_australia","adult","congressional_voting_records","thoraric_surgery","early_stage_diabetes_risk","bank+marketing","online_shoppers_purchasing_intention","seismic_bumps","diabetic_retinopathy_debrecen","south_german_credit","churn","ring","twonorm","car_evaluation","cdc_diabetes_health_indicators","mushroom","agaricus_lepiota","glioma+grading+clinical+and+mutation+features","default_of_credit_card_clients","credit_approval_germany","hypothyroid","steel_plates_faults_Bumps","steel_plates_faults_Dirtiness","steel_plates_faults_K-Scatch","steel_plates_faults_Other-Faults","steel_plates_faults_Pastry","steel_plates_faults_Stains","steel_plates_faults_Z-Scratch","breast_cancer_wisconsin_diagnostic","cervical_cancer_Biopsy","cervical_cancer_Citology","cervical_cancer_Hinselmann","cervical_cancer_Schiller","Mesothelioma's_disease","ionosphere","chess","qsar_biodegradation","spectf","tokyo1","spambase","coil2000","naticusdroid+android+permissions","taiwanese_bankruptcy_prediction","Hill_Valley_with_noise","Hill_Valley_without_noise"]
METHs = ["NBridge1-0.1250000000/relu-4","NBridge1-0.2500000000/relu-4","NBridge1-0.5000000000/relu-4","NBridge1-1.0000000000/relu-4","NBridge1-2.0000000000/relu-4"]
func(DATAs,METHs)
