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
if os.path.isdir("./Plots/plot7")==False: os.makedirs("./Plots/plot7")
color = ["red","green","blue","pink"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
LAMs3 = [10.**(i-5.) for i in range(9)]
SIZEs = [25,50,100,200,400]

def func(TYPEs,METHs,DIMs):
    for DIM,TYPE in itertools.product(DIMs,TYPEs):
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
            res[si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        #
        #bp0 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[0], linewidth=1.5), whiskerprops=dict(color=color[0], linewidth=1.5), capprops=dict(color=color[0], linewidth=1.5), flierprops=dict(markeredgecolor=color[0], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp0['boxes']: b.set_edgecolor(color[0]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[0])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[0])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[0])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[0])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[0])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="ERM", lw=1.5, facecolor='None', edgecolors=color[0], zorder=3)
        # Bridge
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        #bp1 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[1], linewidth=1.5), whiskerprops=dict(color=color[1], linewidth=1.5), capprops=dict(color=color[1], linewidth=1.5), flierprops=dict(markeredgecolor=color[1], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp1['boxes']: b.set_edgecolor(color[1]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[1])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[1])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFBridge", lw=1.5, facecolor='None', edgecolors=color[1], zorder=3)
        # GSLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me+5],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        #bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[2], linewidth=1.5), whiskerprops=dict(color=color[2], linewidth=1.5), capprops=dict(color=color[2], linewidth=1.5), flierprops=dict(markeredgecolor=color[2], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp2['boxes']: b.set_edgecolor(color[2]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[2])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[2])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="^", label="NNSGLasso", lw=1.5, facecolor='None', edgecolors=color[2], zorder=3)
        # GSLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                for se,la1,la2 in itertools.product(range(SEED),range(len(LAMs2)),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f-%.10f/1-seed-%d.csv"%(METHs[me+10],TYPE,DIM,SIZEs[si],LAMs2[la1],LAMs3[la2],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        #bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[2], linewidth=1.5), whiskerprops=dict(color=color[2], linewidth=1.5), capprops=dict(color=color[2], linewidth=1.5), flierprops=dict(markeredgecolor=color[2], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp2['boxes']: b.set_edgecolor(color[2]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[3])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[3])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="x", label="NNSGLasso'", lw=1.5, facecolor='None', edgecolors=color[3], zorder=3)
        #
        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel('n'); plt.ylabel('MCEE'); plt.tight_layout()
        plt.savefig("./Plots/plot7/%s_dim-%d_Closs.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
            res[si,se] = tmp[np.argmin(tmp[:,-3]),-1]
        #
        #bp0 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[0], linewidth=1.5), whiskerprops=dict(color=color[0], linewidth=1.5), capprops=dict(color=color[0], linewidth=1.5), flierprops=dict(markeredgecolor=color[0], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp0['boxes']: b.set_edgecolor(color[0]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[0])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[0])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[0])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[0])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[0])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="s", label="ERM", lw=1.5, facecolor='None', edgecolors=color[0], zorder=3)
        # Bridge
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        #bp1 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[1], linewidth=1.5), whiskerprops=dict(color=color[1], linewidth=1.5), capprops=dict(color=color[1], linewidth=1.5), flierprops=dict(markeredgecolor=color[1], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp1['boxes']: b.set_edgecolor(color[1]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[1])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[1])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFBridge", lw=1.5, facecolor='None', edgecolors=color[1], zorder=3)
        # GSLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me+5],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        #bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[2], linewidth=1.5), whiskerprops=dict(color=color[2], linewidth=1.5), capprops=dict(color=color[2], linewidth=1.5), flierprops=dict(markeredgecolor=color[2], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp2['boxes']: b.set_edgecolor(color[2]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[2])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[2])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="^", label="NNSGLasso", lw=1.5, facecolor='None', edgecolors=color[2], zorder=3)
        # GSLasso2
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5):
                for se,la1,la2 in itertools.product(range(SEED),range(len(LAMs2)),range(len(LAMs3))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f-%.10f/1-seed-%d.csv"%(METHs[me+10],TYPE,DIM,SIZEs[si],LAMs2[la1],LAMs3[la2],se), delimiter=",")
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        #bp2 = ax.boxplot([res[0],res[1],res[2],res[3],res[4]], positions=[0,1,2,3,4], patch_artist=True, widths=0.5, medianprops=dict(color=color[2], linewidth=1.5), whiskerprops=dict(color=color[2], linewidth=1.5), capprops=dict(color=color[2], linewidth=1.5), flierprops=dict(markeredgecolor=color[2], markeredgewidth=1, marker="x"), boxprops={'fill': None}, showfliers=False)
        #for b in bp2['boxes']: b.set_edgecolor(color[2]); b.set_linestyle("--"); b.set_linewidth(1.5); b.set_facecolor(None)
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[3])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[3])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="x", label="NNSGLasso'", lw=1.5, facecolor='None', edgecolors=color[3], zorder=3)
        #
        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel('n'); plt.ylabel('MCR'); plt.tight_layout()
        plt.savefig("./Plots/plot7/%s_dim-%d_miscla.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
#"""
TYPEs = ["non-additive-sq2"]#"non-additive-sq1",
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"   ,
         "NGLasso12-0.0000000000/relu-4"  ,"NGLasso12-0.2500000000/relu-4"  ,"NGLasso12-0.5000000000/relu-4"  ,"NGLasso12-0.7500000000/relu-4"  ,"NGLasso12-1.0000000000/relu-4"  ]
DIMs   = [10]
func(TYPEs,METHs,DIMs)