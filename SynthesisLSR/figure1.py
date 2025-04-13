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
if os.path.isdir("./Figures/figure1")==False: os.makedirs("./Figures/figure1")

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]

def func(TYPEs,METHs,DIMs):
    for DIM,TYPE in itertools.product(DIMs,TYPEs):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
            res[si,se] = np.log(tmp[np.argmin(tmp[:,-2]),-1])
        #
        bp0 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="k", linewidth=1.5), whiskerprops=dict(color="k", linewidth=1.5), capprops=dict(color="k", linewidth=1.5), flierprops=dict(markergecolor="k", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp0['boxes']: b.set_edgecolor("k"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="k")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="ERM", lw=1.5, facecolor='None', edgecolors="k", zorder=3)
        # Bridge
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp1 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="r", linewidth=1.5), whiskerprops=dict(color="r", linewidth=1.5), capprops=dict(color="r", linewidth=1.5), flierprops=dict(markergecolor="r", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor("r"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="r")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFBridge", lw=1.5, facecolor='None', edgecolors="r", zorder=3)
        # GSLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                result = np.zeros((len(LAMs2),SEED))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me+5],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="g", linewidth=1.5), whiskerprops=dict(color="g", linewidth=1.5), capprops=dict(color="g", linewidth=1.5), flierprops=dict(markergecolor="g", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor("g"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="g")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="^", label="NNSGLasso", lw=1.5, facecolor='None', edgecolors="g", zorder=3)
        # Lasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            result = np.zeros((len(LAMs2),SEED))
            for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                tmp = np.loadtxt("./Results/LBridge-1.0000000000/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp3 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="b", linewidth=1.5), whiskerprops=dict(color="b", linewidth=1.5), capprops=dict(color="b", linewidth=1.5), flierprops=dict(markergecolor="b", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor("b"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="b")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="v", label="Lasso", lw=1.5, facecolor='None', edgecolors="b", zorder=3)
        # SpAM
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            result = np.zeros((len(LAMs2),SEED))
            for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                tmp = np.loadtxt("./Results/ABridge-1.0000000000/relu-4-10/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                tmp1[se], tmp2[se] = np.min(tmp[:,-2]), np.log(tmp[np.argmin(tmp[:,-2]),-1])
                if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        bp4 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="c", linewidth=1.5), whiskerprops=dict(color="c", linewidth=1.5), capprops=dict(color="c", linewidth=1.5), flierprops=dict(markergecolor="c", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor("c"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="c")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="<", label="SpAM", lw=1.5, facecolor='None', edgecolors="c", zorder=3)
        # random forest
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/RandomForest/%s/dim-%d/size-%d/seed-%d.csv"%(TYPE,DIM,SIZEs[si],se), delimiter=",")
            res[si,se] = np.log(tmp[-1])
        #
        bp5 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="gold", linewidth=1.5), whiskerprops=dict(color="gold", linewidth=1.5), capprops=dict(color="gold", linewidth=1.5), flierprops=dict(markergecolor="gold", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp5['boxes']: b.set_edgecolor("gold"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="gold")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker=">", label="random forest", lw=1.5, facecolor='None', edgecolors="gold", zorder=3)
        # oracle
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,3,SIZEs[si],0.0,se), delimiter=",")
            res[si,se] = np.log(tmp[np.argmin(tmp[:,-2]),-1])
        #
        bp6 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color="m", linewidth=1.5), whiskerprops=dict(color="m", linewidth=1.5), capprops=dict(color="m", linewidth=1.5), flierprops=dict(markergecolor="m", markergewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp6['boxes']: b.set_edgecolor("m"); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color="m")
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="D", label="oracle", lw=1.5, facecolor='None', edgecolors="m", zorder=3)
        #
        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel(r'$n$'); plt.ylabel(r'log(test MSE)'); plt.tight_layout()
        plt.savefig("./Figures/figure1/%s_dim-%d_Rloss.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]
METHs = ["NBridge-0.1250000000/relu-4-10","NBridge-0.2500000000/relu-4-10","NBridge-0.5000000000/relu-4-10","NBridge-1.0000000000/relu-4-10","NBridge-2.0000000000/relu-4-10",
         "NGLasso-0.0000000000/relu-4"   ,"NGLasso-0.2500000000/relu-4"   ,"NGLasso-0.5000000000/relu-4"   ,"NGLasso-0.7500000000/relu-4"   ,"NGLasso-1.0000000000/relu-4"   ]
DIMs   = [10]
func(TYPEs,METHs,DIMs)
#
TYPEs = ["non-additive-sq2"]
METHs = ["NBridge-0.1250000000/relu-4-10","NBridge-0.2500000000/relu-4-10","NBridge-0.5000000000/relu-4-10","NBridge-1.0000000000/relu-4-10","NBridge-2.0000000000/relu-4-10",
         "NGLasso-0.0000000000/relu-4"   ,"NGLasso-0.2500000000/relu-4"   ,"NGLasso-0.5000000000/relu-4"   ,"NGLasso-0.7500000000/relu-4"   ,"NGLasso-1.0000000000/relu-4"   ]
DIMs  = [5,20]
func(TYPEs,METHs,DIMs)