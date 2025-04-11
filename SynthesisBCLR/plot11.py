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
if os.path.isdir("./Plots/plot11")==False: os.makedirs("./Plots/plot11")
color = ["red","green","blue","cyan","gold","magenta","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
#"""
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # GSLasso, Bridge
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-4])
        fs = np.zeros((len(METHs),DIM,SEED))
        for me in range(len(METHs)):
            la = np.argmin(np.mean(res[me,:,:], axis=1))
            for se in range(SEED):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                fs[me,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]
                fs[me,:,se] /= np.max(fs[me,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp2 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0.25$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp3 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0.5$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp4 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0.75$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp5 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=1$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test risk
        plt.savefig("./Plots/plot11/NNSGLasso_%s_dim-%d_size-%d_Closs.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # GSLasso, Bridge
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-3])
        fs = np.zeros((len(METHs),DIM,SEED))
        for me in range(len(METHs)):
            la = np.argmin(np.mean(res[me,:,:], axis=1))
            for se in range(SEED):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                fs[me,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]
                fs[me,:,se] /= np.max(fs[me,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp2 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0.25$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp3 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0.5$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp4 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=0.75$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp5 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$\alpha=1$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test misclassification rate
        plt.savefig("./Plots/plot11/NNSGLasso_%s_dim-%d_size-%d_miscla.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"]
DIMs   = [10]
func(TYPEs,METHs,DIMs)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"]
DIMs  = [5,20]
func(TYPEs,METHs,DIMs)

#"""
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # GSLasso, Bridge
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-4])
        fs = np.zeros((len(METHs),DIM,SEED))
        for me in range(len(METHs)):
            la = np.argmin(np.mean(res[me,:,:], axis=1))
            for se in range(SEED):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                fs[me,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]
                fs[me,:,se] /= np.max(fs[me,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=0.125$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp2 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=0.25$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp3 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=0.5$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp4 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=1$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp5 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=2$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test risk
        plt.savefig("./Plots/plot11/MFBridge_%s_dim-%d_size-%d_Closs.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # GSLasso, Bridge
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-3])
        fs = np.zeros((len(METHs),DIM,SEED))
        for me in range(len(METHs)):
            la = np.argmin(np.mean(res[me,:,:], axis=1))
            for se in range(SEED):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                fs[me,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]
                fs[me,:,se] /= np.max(fs[me,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=0.125$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp2 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=0.25$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp3 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=0.5$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp4 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=1$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.15*(k-2)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp5 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"$q=2$", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test misclassification rate
        plt.savefig("./Plots/plot11/MFBridge_%s_dim-%d_size-%d_miscla.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
DIMs   = [10]
func(TYPEs,METHs,DIMs)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
DIMs  = [5,20]
func(TYPEs,METHs,DIMs)

#"""
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # GSLasso, Bridge
        fs = np.zeros((len(METHs),DIM,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],se), delimiter=",")
            fs[me,:,se] = tmp[2:-4]
            fs[me,:,se] /= np.max(fs[me,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test risk
        plt.savefig("./Plots/plot11/RandomForest_%s_dim-%d_size-%d_Closs.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); #plt.yscale('log')
        # GSLasso, Bridge
        fs = np.zeros((len(METHs),DIM,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],se), delimiter=",")
            fs[me,:,se] = tmp[2:-4]
            fs[me,:,se] /= np.max(fs[me,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        #bp1 = sns.violinplot(data=[fs[k,i,:] for i in range(DIM)], density_norm="width", fill=False, width=0.8, cut=0, linewidth=1, inner=None, color=color[k], linecolor=color[k])
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        #ax.legend()#loc='lower right'
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n_{\rm tra}\!=\!%d$'%SIZEs[si]); plt.tight_layout()#test misclassification rate
        plt.savefig("./Plots/plot11/RandomForest_%s_dim-%d_size-%d_miscla.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["RandomForest"]
DIMs   = [10]
func(TYPEs,METHs,DIMs)
#
TYPEs = ["non-additive-sq2"]#["linear","additive-sq","non-additive-sq1","non-additive-sq2","non-additive-sq3","additive-sin","non-additive-sin1","non-additive-sin2","non-additive-sin3"]
METHs = ["RandomForest"]
DIMs  = [5,20]
func(TYPEs,METHs,DIMs)