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
if os.path.isdir("./Plots/plot3")==False: os.makedirs("./Plots/plot3")
color = ["red","green","blue","cyan","magenta","gold","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [25,50,100,200,400]
#==================================================#
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        fs = np.zeros((len(METHs),DIM,len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            fs[me,:,la,se] = tmp[np.argmin(tmp[:,-2]),3:-2]
        #
        k = 0
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        abc = np.zeros((len(METHs),3,SEED))
        for me,se in itertools.product(range(len(METHs)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
            abc[me,0,se] = np.min(tmp[:,-2])
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
            abc[me,1,se] = tmp[np.argmin(tmp[:,-2]),1]
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[2],TYPE,DIM,SIZEs[si],LAMs2[-1],se), delimiter=",")
            abc[me,2,se] = np.min(tmp[:,-2])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        for me  in range(len(METHs)):
            serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
            plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([4*i for i in range(9)],[-5,-4,-3,-2,-1,0,1,2,3]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'$s_j$, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot3/MFBridge_%s_dim-%d_size-%d_Rloss.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
DIMs   = [10,5,20]#
# func(TYPEs,METHs,DIMs)
#==================================================#
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        #
        fs = np.zeros((len(METHs),DIM,len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            fs[me,:,la,se] = tmp[np.argmin(tmp[:,-2]),3:-2]
        #
        k = 0
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="o", zorder=3)
        #
        k = 1
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="^", zorder=3)
        #
        k = 2
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        #
        k = 3
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker="<", zorder=3)
        #
        k = 4
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), lw=1.5, color=color[k], linestyle='dotted')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,0,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashed')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,1,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), lw=1.5, color=color[k], linestyle='dashdot')
        ax.scatter(np.arange(len(LAMs1)), np.mean(fs[k,2,:,:], axis=1), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), lw=1.5, color=color[k], linestyle='solid')
        ax.scatter(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,3:,:,:], axis=2), axis=0), s=60, lw=1.5, facecolor='None', edgecolors=color[k], marker=">", zorder=3)
        #
        # abc = np.zeros((len(METHs),3,SEED))
        # for me,se in itertools.product(range(len(METHs)),range(SEED)):
        #     tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%("NGLasso1-0.0000000000/relu-4",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
        #     abc[me,0,se] = np.min(tmp[:,-2])
        #     tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
        #     abc[me,1,se] = tmp[np.argmin(tmp[:,-2]),1]
        #     tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[-1],se), delimiter=",")
        #     abc[me,2,se] = np.min(tmp[:,-2])
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        # for me  in range(len(METHs)):
        #     serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
        #     plt.plot(np.ones(10)*4*(np.log10(serachlam)+5), np.linspace(BOT,TOP,10), color=color[me], lw=1.5, linestyle="--", zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([4*i for i in range(9)],[-5,-4,-3,-2,-1,0,1,2,3]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'$s_j$, $n=%d$'%SIZEs[si]); plt.tight_layout()
        plt.savefig("./Plots/plot3/NNSGLasso_%s_dim-%d_size-%d_Rloss.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4"]
DIMs   = [10,5,20]#
func(TYPEs,METHs,DIMs)
