import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
os.makedirs("./Plots/plot3", exist_ok=True)
color = ["red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown"]
marker = ["D","o","P","X","^","v","<",">","*","p"]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--size', type=int)
parser.add_argument('--j', type=int)
args = parser.parse_args()
DATA = args.data
SIZE = args.size
J = args.j

SEED  = 100
LAMs1 = [i/8.-6. for i in range(81)]
LAMs2 = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [SIZE]
DIMs   = [10]
TYPEs = [DATA]
METHs = [["1MFBridge-0.1250000000/relu-4-20","1MFBridge-0.2500000000/relu-4-20","1MFBridge-0.5000000000/relu-4-20","1MFBridge-1.0000000000/relu-4-20","1MFBridge-2.0000000000/relu-4-20"],
         ["1MFElim-0.1250000000/relu-4-20","1MFElim-0.2500000000/relu-4-20","1MFElim-0.5000000000/relu-4-20","1MFElim-1.0000000000/relu-4-20","1MFElim-2.0000000000/relu-4-20"],
         ["1MFLoss-0.1250000000/relu-4-20","1MFLoss-0.2500000000/relu-4-20","1MFLoss-0.5000000000/relu-4-20","1MFLoss-1.0000000000/relu-4-20","1MFLoss-2.0000000000/relu-4-20"],
         ["1MFRisk-0.1250000000/relu-4-20","1MFRisk-0.2500000000/relu-4-20","1MFRisk-0.5000000000/relu-4-20","1MFRisk-1.0000000000/relu-4-20","1MFRisk-2.0000000000/relu-4-20"],
         ["1Bridge-0.1250000000","1Bridge-0.2500000000","1Bridge-0.5000000000","1Bridge-1.0000000000","1Bridge-2.0000000000"],
         ["1SpAM-0.1250000000/relu-4","1SpAM-0.2500000000/relu-4","1SpAM-0.5000000000/relu-4","1SpAM-1.0000000000/relu-4","1SpAM-2.0000000000/relu-4"],
         ["1SPINN-0.0000000000/relu-4","1SPINN-0.2500000000/relu-4","1SPINN-0.5000000000/relu-4","1SPINN-0.7500000000/relu-4","1SPINN-1.0000000000/relu-4"],
         ["1LassoNet-0.1000000000/relu-4","1LassoNet-0.3162280000/relu-4","1LassoNet-1.0000000000/relu-4","1LassoNet-3.1622780000/relu-4","1LassoNet-10.0000000000/relu-4"],
         ["1STG-0.1250000000/relu-4","1STG-0.2500000000/relu-4","1STG-0.5000000000/relu-4","1STG-1.0000000000/relu-4","1STG-2.0000000000/relu-4"]][J]

def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.grid(True)

        fs = np.zeros((len(METHs),DIM,len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            fs[me,:,la,se] = tmp[np.argmin(tmp[:,-4]),4:-4]

        k = 0
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0], axis=1), lw=1., color=color[k], linestyle='dotted')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1], axis=1), lw=1., color=color[k], linestyle='dashed')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2], axis=1), lw=1., color=color[k], linestyle='dashdot')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,3], axis=1), lw=1., color=color[k], linestyle='solid')
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,4:], axis=0), axis=1), lw=2., alpha=0.5, color=color[k], linestyle='solid')

        k = 1
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0], axis=1), lw=1., color=color[k], linestyle='dotted')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1], axis=1), lw=1., color=color[k], linestyle='dashed')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2], axis=1), lw=1., color=color[k], linestyle='dashdot')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,3], axis=1), lw=1., color=color[k], linestyle='solid')
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,4:], axis=0), axis=1), lw=2., alpha=0.5, color=color[k], linestyle='solid')

        k = 2
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0], axis=1), lw=1., color=color[k], linestyle='dotted')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1], axis=1), lw=1., color=color[k], linestyle='dashed')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2], axis=1), lw=1., color=color[k], linestyle='dashdot')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,3], axis=1), lw=1., color=color[k], linestyle='solid')
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,4:], axis=0), axis=1), lw=2., alpha=0.5, color=color[k], linestyle='solid')

        k = 3
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0], axis=1), lw=1., color=color[k], linestyle='dotted')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1], axis=1), lw=1., color=color[k], linestyle='dashed')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2], axis=1), lw=1., color=color[k], linestyle='dashdot')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,3], axis=1), lw=1., color=color[k], linestyle='solid')
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,4:], axis=0), axis=1), lw=2., alpha=0.5, color=color[k], linestyle='solid')

        k = 4
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,0], axis=1), lw=1., color=color[k], linestyle='dotted')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,1], axis=1), lw=1., color=color[k], linestyle='dashed')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,2], axis=1), lw=1., color=color[k], linestyle='dashdot')
        ax.plot(np.arange(len(LAMs1)), np.mean(fs[k,3], axis=1), lw=1., color=color[k], linestyle='solid')
        ax.plot(np.arange(len(LAMs1)), np.mean(np.mean(fs[k,4:], axis=0), axis=1), lw=2., alpha=0.5, color=color[k], linestyle='solid')

        if METHs[0].split("-")[0]=="1MFBridge":
            abc = np.zeros((len(METHs),3,SEED))
            for me,se in itertools.product(range(len(METHs)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",TYPE,DIM,SIZEs[si],0,se), delimiter=",")
                abc[me,0,se] = tmp[np.argmin(tmp[:,-4]),0]
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
                abc[me,1,se] = tmp[np.argmin(tmp[:,-4]),1]
                tmp = np.loadtxt("./Results/1Const/%s/dim-%d/size-%d/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],se), delimiter=",")
                abc[me,2,se] = tmp[2]
            BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
            for me  in range(len(METHs)):
                serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
                plt.plot(np.ones(10)*8*(np.log10(serachlam)+6), np.linspace(BOT,TOP,10), color=color[me], lw=1., alpha=0.5, zorder=1)

        plt.ylim([BOT,TOP]); plt.xticks([8*i for i in range(11)],[-6,-5,-4,-3,-2,-1,0,1,2,3,4]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'FS for classification, $N=%d$'%SIZEs[si], y=0.45); plt.tight_layout()
        plt.savefig("./Plots/plot3/%s_%s_%d_%d_CFS.png"%(METHs[0].split("-")[0],TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

func(TYPEs,METHs,DIMs)
