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
if os.path.isdir("./Figures/figure2")==False: os.makedirs("./Figures/figure2")
color = ["red","green","blue","cyan","gold","magenta","black"]

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
#==================================================#
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-4]),-2]
        #
        k = 0
        bp1 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.125$", lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.25$", lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.5$", lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=1$", lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=2$", lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%("NGLasso-0.0000000000/relu-4",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[2],TYPE,DIM,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([i*4 for i in range(7)],[-5,-4,-3,-2,-1,0,1]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCEE, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Figures/figure2/MFBridge_%s_dim-%d_size-%d_Closs.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-3]),-1]
        #
        k = 0
        bp1 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.125$", lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.25$", lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=0.5$", lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=1$", lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$q=2$", lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%("NGLasso-0.0000000000/relu-4",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[2],TYPE,DIM,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([i*4 for i in range(7)],[-5,-4,-3,-2,-1,0,1]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCR, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Figures/figure2/MFBridge_%s_dim-%d_size-%d_miscla.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]
METHs = ["NBridge-0.1250000000/relu-4-10","NBridge-0.2500000000/relu-4-10","NBridge-0.5000000000/relu-4-10","NBridge-1.0000000000/relu-4-10","NBridge-2.0000000000/relu-4-10"]
DIMs   = [10]#5,20
func(TYPEs,METHs,DIMs)
#==================================================#
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-4]),-2]
        #
        k = 0
        bp1 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0$", lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.25$", lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.5$", lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.75$", lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=1$", lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%("NGLasso-0.0000000000/relu-4",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([i*4 for i in range(7)],[-5,-4,-3,-2,-1,0,1]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCEE, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Figures/figure2/NNSGLasso_%s_dim-%d_size-%d_Closs.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        res = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = tmp[np.argmin(tmp[:,-3]),-1]
        #
        k = 0
        bp1 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0$", lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.25$", lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.5$", lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=0.75$", lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([res[k,i,:] for i in range(len(LAMs1))], positions=np.arange(len(LAMs1)), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.plot(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), lw=1.5, color=color[k])
        ax.scatter(np.arange(len(LAMs1)), np.mean(res[k,:,:], axis=1), s=60, label=r"$\alpha=1$", lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%("NGLasso-0.0000000000/relu-4",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-4])
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-4])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([i*4 for i in range(7)],[-5,-4,-3,-2,-1,0,1]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'test MCR, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Figures/figure2/NNSGLasso_%s_dim-%d_size-%d_miscla.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]
METHs = ["NGLasso-0.0000000000/relu-4"   ,"NGLasso-0.2500000000/relu-4"   ,"NGLasso-0.5000000000/relu-4"   ,"NGLasso-0.7500000000/relu-4"   ,"NGLasso-1.0000000000/relu-4"]
DIMs   = [10]#5,20
func(TYPEs,METHs,DIMs)