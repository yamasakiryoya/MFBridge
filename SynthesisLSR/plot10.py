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
if os.path.isdir("./Plots/plot10")==False: os.makedirs("./Plots/plot10")
color = ["red","green","blue","cyan","gold","magenta","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
#"""
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        fs = np.zeros((3,DIM,SEED))
        # GSLasso
        res = np.zeros((5,len(LAMs2),SEED))
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-2])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[0,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[0,:,se] /= np.max(fs[0,:,se])
        # MFBridge
        res = np.zeros((5,len(LAMs2),SEED))
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me+5],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-2])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]+5],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[1,:,se] = tmp[np.argmin(tmp[:,-2]),3:-2]; fs[1,:,se] /= np.max(fs[1,:,se])
        # random forest
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/1-seed-%d.csv"%(METHs[10],TYPE,DIM,SIZEs[si],se), delimiter=",")
            fs[2,:,se] = tmp[2:-2]; fs[2,:,se] /= np.max(fs[2,:,se])

        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#-0.3
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"NNSGLasso", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp2 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"MFBridge", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.3
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp3 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test risk#
        plt.savefig("./Plots/plot10/%s_dim-%d_size-%d_Rloss.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
        "RandomForest"]
DIMs   = [10]
func(TYPEs,METHs,DIMs)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
        "RandomForest"]
DIMs  = [5,20]
#func(TYPEs,METHs,DIMs)