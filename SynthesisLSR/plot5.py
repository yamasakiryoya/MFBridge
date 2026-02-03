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
if os.path.isdir("./Plots/plot5")==False: os.makedirs("./Plots/plot5")
color = ["red","green","blue","cyan","magenta","gold","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [100]#[25,50,100,200,400]

def func(TYPEs,METHs,METHs2,DIMs):
    for me,DIM,TYPE,si in itertools.product(range(len(METHs)),DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True); ay = ax.twinx()
        #
        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-2])
        optla = LAMs2[np.argmin(np.mean(res[la,:]))]
        LC = np.zeros((2000,4,20))
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],optla,se), delimiter=",")
            LC[:,0,se] = np.log(tmp[:,0])
            LC[:,1,se] = np.log(tmp[:,-2])
            LC[:,2,se] = tmp[:,1]
        #
        ax.plot(range(1,2001), np.mean(LC[:,0,:], axis=1), lw=1.5, color=color[0], linestyle='dotted')
        ax.fill_between(range(1,2001), np.quantile(LC[:,0,:], axis=1, q=0.25), np.quantile(LC[:,0,:], axis=1, q=0.75), color=color[0], alpha=0.3, edgecolor=None)
        ax.plot(range(1,2001), np.mean(LC[:,1,:], axis=1), lw=1.5, color=color[1], linestyle='dashed')
        ax.fill_between(range(1,2001), np.quantile(LC[:,1,:], axis=1, q=0.25), np.quantile(LC[:,1,:], axis=1, q=0.75), color=color[1], alpha=0.3, edgecolor=None)
        ay.plot(range(1,2001), np.mean(LC[:,2,:], axis=1), lw=1.5, color=color[2], linestyle='dashdot')
        ay.fill_between(range(1,2001), np.quantile(LC[:,2,:], axis=1, q=0.25), np.quantile(LC[:,2,:], axis=1, q=0.75), color=color[2], alpha=0.3, edgecolor=None)
        #
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        ax.plot(1+np.ones(10)*np.argmin(np.mean(LC[:,1,:], axis=1)), np.linspace(BOT,TOP,10), color="k", lw=1.5, linestyle="--", zorder=3)
        #
        ax.set_ylim(BOT,TOP); ax.set_xlabel('epoch'); ax.set_ylabel(r'log(MSE)'); ay.set_ylabel(r'regularizer'); plt.tight_layout()
        plt.savefig("./Plots/plot5/%s_%s_dim-%d_size-%d_Rloss.png"%(METHs2[me],TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
TYPEs = ["non-additive-sq"]
METHs = ["NGLasso1-0.0000000000/relu-4"   ,"NGLasso1-0.2500000000/relu-4"   ,"NGLasso1-0.5000000000/relu-4"   ,"NGLasso1-0.7500000000/relu-4"   ,"NGLasso1-1.0000000000/relu-4",
         "NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10"]
METHs2= ["NGLasso1-0.0000000000"   ,"NGLasso1-0.2500000000"   ,"NGLasso1-0.5000000000"   ,"NGLasso1-0.7500000000"   ,"NGLasso1-1.0000000000",
         "NBridge1-0.1250000000-10","NBridge1-0.2500000000-10","NBridge1-0.5000000000-10","NBridge1-1.0000000000-10","NBridge1-2.0000000000-10"]
DIMs   = [10,5,20]#
func(TYPEs,METHs,METHs2,DIMs)
