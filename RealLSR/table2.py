import os
import itertools
import numpy as np
from scipy import stats

SEED  = 20
DATAs = ["nikuradse_2","nikuradse_1","vinnie","analcatdata_apnea1","analcatdata_apnea2","visualizing_galaxy","ESL","synchronous_machine","ERA","LEV","pollen","combined_cycle_power_plant","airfoil_self-noise","yacht_hydrodynamics","real_estate_valuation","qsar_fish_toxicity","auto-mpg","no2","pm10","auction_verification_time","qsar_aquatic_toxicity","energy_efficiency_Cooling-Load","energy_efficiency_Heating-Load","concrete_compressive_strength","abalone","puma8NH","houses","house_8L","echoMonths","lowbwt","CASP","breastTumor","rmftsa_ladata","SWD","solar_flare","2dplanes","fried","mv","pwLinear","pharynx","poker","Bike_Sharing_day","wine_quality_red","wine_quality_white","forestfires","cpu_small","Bike_Sharing_hour","wind","sgemm_product_Run1","sgemm_product_Run2","sgemm_product_Run3","sgemm_product_Run4","house_16H","pbc","cpu_act","student_mat_G1","student_mat_G2","student_mat_G3","student_por_G1","student_por_G2","student_por_G3","satellite_image","pol","online_news_popularity","superconductivty"]
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]

resM_MSE  = np.zeros((len(DATAs),SEED))
resMB_MSE = np.zeros((len(DATAs),SEED))
resNB_MSE = np.zeros((len(DATAs),SEED))
resL1_MSE  = np.zeros((len(DATAs),SEED))
resL2_MSE  = np.zeros((len(DATAs),SEED))
resA1_MSE  = np.zeros((len(DATAs),SEED))
resA2_MSE  = np.zeros((len(DATAs),SEED))
resR1_MSE  = np.zeros((len(DATAs),SEED))
resR2_MSE  = np.zeros((len(DATAs),SEED))
resE_MSE  = np.zeros((len(DATAs),SEED))

#vs N-
for ALP in [0.9,0.95,0.99]:
    print(ALP)
    for d in range(len(DATAs)):
        DATA = DATAs[d]
        #
        Qs = [0.125,0.25,0.5,1,2]
        tmpM_MSE_va = np.zeros((SEED,len(Qs),len(LAMs2)))
        tmpM_MSE_te = np.zeros((SEED,len(Qs),len(LAMs2)))
        for se in range(SEED):
            for q in range(len(Qs)):
                for la in range(len(LAMs2)):
                    tmp = np.loadtxt("./Results/NBridge1-%.10f/relu-4-10/%s/size-100/lam-%.10f/1-seed-%d.csv"%(Qs[q],DATA,LAMs2[la],se), delimiter=",")
                    tmpM_MSE_va[se,q,la] = np.min(tmp[:,-2])
                    tmpM_MSE_te[se,q,la] = tmp[np.argmin(tmp[:,-2]),-1]
            # MFBridge
            idx = np.unravel_index(np.argmin(tmpM_MSE_va[se]), tmpM_MSE_va[se].shape)
            resM_MSE[d,se] = tmpM_MSE_te[se,idx[0],idx[1]]
        # MFBridge (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/NBridge2-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resMB_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # ERM (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/NBridge3-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resE_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # NNSGLasso (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/NGLasso2-%.10f/relu-4/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resNB_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # Lasso () with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/LBridge2-%.10f/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resL1_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # Lasso (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/LBridge3-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resL2_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # SpAM () with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/ABridge2-%.10f/relu-4/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resA1_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # SpAM (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/ABridge3-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resA2_MSE[d,se]   = tmp[np.argmin(tmp[:,-2]),-1]
        # Random forest () with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest2-%.10f/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resR1_MSE[d,se]  = tmp[-1]
        # Random forest (NN) with explicit feature selection
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest3-%.10f/relu-4-10/%s/size-100/1-seed-%d.csv"%(ALP,DATA,se), delimiter=",")
            resR2_MSE[d,se]  = tmp[np.argmin(tmp[:,-2]),-1]


    def test(p):
        print("MSE",p)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resM_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resE_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resNB_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resL1_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resL2_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resA1_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resA2_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resR1_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #
        tmp = np.zeros(3)
        for d in range(len(DATAs)):
            test = stats.mannwhitneyu(resMB_MSE[d], resR2_MSE[d], alternative='less')
            if   test.pvalue<=p:   tmp[0]+=1
            elif test.pvalue>=1-p: tmp[2]+=1
            else:                  tmp[1]+=1
        print(tmp)
        #


    # test(0.1)
    test(0.05)
    # test(0.1/len(DATAs))
    # test(0.05/len(DATAs))