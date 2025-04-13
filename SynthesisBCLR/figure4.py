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
if os.path.isdir("./Figures/figure4")==False: os.makedirs("./Figures/figure4")
color = ["red","green","blue","cyan","gold","magenta","black"]

SEED  = 20
LAMs1 = [i/4.-5. for i in range(25)]
LAMs2 = [10.**(i/4.-5.) for i in range(25)]
SIZEs = [25,50,100,200,400]
#
def func(TYPEs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        fs = np.zeros((5,DIM,SEED))
        # MFBridge
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NBridge-0.1250000000/relu-4-10","NBridge-0.2500000000/relu-4-10","NBridge-0.5000000000/relu-4-10","NBridge-1.0000000000/relu-4-10","NBridge-2.0000000000/relu-4-10"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[0,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[0,:,se] /= np.max(fs[0,:,se])
        # GSLasso
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NGLasso-0.0000000000/relu-4","NGLasso-0.2500000000/relu-4","NGLasso-0.5000000000/relu-4","NGLasso-0.7500000000/relu-4","NGLasso-1.0000000000/relu-4"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[1,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[1,:,se] /= np.max(fs[1,:,se])
        # Lasso
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/LBridge-1.0000000000/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/LBridge-1.0000000000/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[2,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[2,:,se] /= np.max(fs[2,:,se])
        # SpAM
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/ABridge-1.0000000000/relu-4-10/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/ABridge-1.0000000000/relu-4-10/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[3,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]; fs[3,:,se] /= np.max(fs[3,:,se])
        # random forest
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest/%s/dim-%d/size-%d/seed-%d.csv"%(TYPE,DIM,SIZEs[si],se), delimiter=",")
            fs[4,:,se] = tmp[2:-4]; fs[4,:,se] /= np.max(fs[4,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#-0.3
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"MFBridge", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"NNSGLasso", facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"Lasso", facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"SpAM", facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.3
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Figures/figure4/%s_dim-%d_size-%d_Closs.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        fs = np.zeros((5,DIM,SEED))
        # MFBridge
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NBridge-0.1250000000/relu-4-10","NBridge-0.2500000000/relu-4-10","NBridge-0.5000000000/relu-4-10","NBridge-1.0000000000/relu-4-10","NBridge-2.0000000000/relu-4-10"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[0,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[0,:,se] /= np.max(fs[0,:,se])
        # GSLasso
        res = np.zeros((5,len(LAMs2),SEED))
        METHs = ["NGLasso-0.0000000000/relu-4","NGLasso-0.2500000000/relu-4","NGLasso-0.5000000000/relu-4","NGLasso-0.7500000000/relu-4","NGLasso-1.0000000000/relu-4"]
        for me,la,se in itertools.product(range(5),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[me,la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
            fs[1,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[1,:,se] /= np.max(fs[1,:,se])
        # Lasso
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/LBridge-1.0000000000/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/LBridge-1.0000000000/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[2,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[2,:,se] /= np.max(fs[2,:,se])
        # SpAM
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/ABridge-1.0000000000/relu-4-10/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-3])
        for se in range(SEED):
            idx = np.argmin(res[:,se])
            tmp = np.loadtxt("./Results/ABridge-1.0000000000/relu-4-10/%s/dim-%d/size-%d/lam-%.10f/seed-%d.csv"%(TYPE,DIM,SIZEs[si],LAMs2[idx],se), delimiter=",")
            fs[3,:,se] = tmp[np.argmin(tmp[:,-3]),4:-4]; fs[3,:,se] /= np.max(fs[3,:,se])
        # random forest
        for se in range(SEED):
            tmp = np.loadtxt("./Results/RandomForest/%s/dim-%d/size-%d/seed-%d.csv"%(TYPE,DIM,SIZEs[si],se), delimiter=",")
            fs[4,:,se] = tmp[2:-4]; fs[4,:,se] /= np.max(fs[4,:,se])
        #
        k = 0
        bp1 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#-0.3
        for b in bp1['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"MFBridge", facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        bp2 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp2['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"NNSGLasso", facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        bp3 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp3['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"Lasso", facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        bp4 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        for b in bp4['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"SpAM", facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        bp5 = ax.boxplot([fs[k,i,:] for i in range(DIM)], positions=np.arange(DIM), patch_artist=True, widths=0.5, medianprops=dict(color=color[k], linewidth=1.5), whiskerprops=dict(color=color[k], linewidth=1.5), capprops=dict(color=color[k], linewidth=1.5), flierprops=dict(markeredgecolor=color[k], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)#+0.3
        for b in bp5['boxes']: b.set_edgecolor(color[k]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1.5, label=r"random forest", facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$s_j/\max(s_k)$, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Figures/figure4/%s_dim-%d_size-%d_miscla.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq2"]
DIMs   = [10]#5,20
func(TYPEs,DIMs)
