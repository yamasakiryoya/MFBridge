import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
os.makedirs("./Plots/plot5", exist_ok=True)
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
         ["1SPINN-0.0000000000/relu-4","1SPINN-0.2500000000/relu-4","1SPINN-0.5000000000/relu-4","1SPINN-0.7500000000/relu-4","1SPINN-1.0000000000/relu-4"],
         ["1LassoNet-0.1000000000/relu-4","1LassoNet-0.3162280000/relu-4","1LassoNet-1.0000000000/relu-4","1LassoNet-3.1622780000/relu-4","1LassoNet-10.0000000000/relu-4"],
         ["1STG-0.1250000000/relu-4","1STG-0.2500000000/relu-4","1STG-0.5000000000/relu-4","1STG-1.0000000000/relu-4","1STG-2.0000000000/relu-4"],
         ["1MFElim-0.1250000000/relu-4-20","1MFElim-0.2500000000/relu-4-20","1MFElim-0.5000000000/relu-4-20","1MFElim-1.0000000000/relu-4-20","1MFElim-2.0000000000/relu-4-20"],
         ["1MFLoss-0.1250000000/relu-4-20","1MFLoss-0.2500000000/relu-4-20","1MFLoss-0.5000000000/relu-4-20","1MFLoss-1.0000000000/relu-4-20","1MFLoss-2.0000000000/relu-4-20"],
         ["1MFRisk-0.1250000000/relu-4-20","1MFRisk-0.2500000000/relu-4-20","1MFRisk-0.5000000000/relu-4-20","1MFRisk-1.0000000000/relu-4-20","1MFRisk-2.0000000000/relu-4-20"],
         ["1Bridge-0.1250000000","1Bridge-0.2500000000","1Bridge-0.5000000000","1Bridge-1.0000000000","1Bridge-2.0000000000"],
         ["1SpAM-0.1250000000/relu-4","1SpAM-0.2500000000/relu-4","1SpAM-0.5000000000/relu-4","1SpAM-1.0000000000/relu-4","1SpAM-2.0000000000/relu-4"]][J]

def func(TYPEs,METHs,DIMs):
    for me,DIM,TYPE,si in itertools.product(range(len(METHs)),DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.grid(True); ay = ax.twinx()

        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        optla = LAMs2[np.argmin(np.mean(res[la,:]))]
        LC = np.zeros((200,4,SEED))
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],optla,se), delimiter=",")
            LC[:,0,se] = tmp[:,0]
            LC[:,1,se] = tmp[:,-4]
            LC[:,2,se] = tmp[:,1]

        ax.plot(range(1,201), np.mean(LC[:,0,:], axis=1), lw=1.5, color=color[0], linestyle='dotted')
        ax.fill_between(range(1,201), np.quantile(LC[:,0,:], axis=1, q=0.25), np.quantile(LC[:,0,:], axis=1, q=0.75), color=color[0], alpha=0.3, edgecolor=None)
        ax.plot(range(1,201), np.mean(LC[:,1,:], axis=1), lw=1.5, color=color[1], linestyle='dashed')
        ax.fill_between(range(1,201), np.quantile(LC[:,1,:], axis=1, q=0.25), np.quantile(LC[:,1,:], axis=1, q=0.75), color=color[1], alpha=0.3, edgecolor=None)
        ay.plot(range(1,201), np.mean(LC[:,2,:], axis=1), lw=1.5, color=color[2], linestyle='dashdot')
        ay.fill_between(range(1,201), np.quantile(LC[:,2,:], axis=1, q=0.25), np.quantile(LC[:,2,:], axis=1, q=0.75), color=color[2], alpha=0.3, edgecolor=None)

        ax.set_ylim(0,3)
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        ax.plot(1+np.ones(10)*np.argmin(np.mean(LC[:,1,:], axis=1)), np.linspace(BOT,TOP,10), color="k", lw=1., alpha=0.5, zorder=3)

        ax.set_ylim(BOT,TOP); ax.set_xlabel('epoch'); ax.set_ylabel(r'MCE'); ay.set_ylabel(r'regularizer'); plt.tight_layout()
        plt.savefig("./Plots/plot5/%s_%s_%d_%d_CMCE.png"%(METHs[me].split("/")[0],TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

func(TYPEs,METHs,DIMs)

def func(TYPEs,METHs,DIMs):
    for me,DIM,TYPE,si in itertools.product(range(len(METHs)),DIMs,TYPEs,range(len(SIZEs))):
        os.makedirs("./Plots/plot5/%s"%TYPE, exist_ok=True)

        fig = plt.figure(figsize=(8,5)); ax = fig.add_subplot(111); ax.grid(True); ay = ax.twinx()

        res = np.zeros((len(LAMs2),SEED))
        for la,se in itertools.product(range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res[la,se] = np.min(tmp[:,-4])
        optla = LAMs2[np.argmin(np.mean(res[la,:]))]
        LC = np.zeros((200,4,SEED))
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],optla,se), delimiter=",")
            LC[:,0,se] = tmp[:,3]
            LC[:,1,se] = tmp[:,-3]
            LC[:,2,se] = tmp[:,1]

        ax.plot(range(1,201), np.mean(LC[:,0,:], axis=1), lw=1.5, color=color[0], linestyle='dotted')
        ax.fill_between(range(1,201), np.quantile(LC[:,0,:], axis=1, q=0.25), np.quantile(LC[:,0,:], axis=1, q=0.75), color=color[0], alpha=0.3, edgecolor=None)
        ax.plot(range(1,201), np.mean(LC[:,1,:], axis=1), lw=1.5, color=color[1], linestyle='dashed')
        ax.fill_between(range(1,201), np.quantile(LC[:,1,:], axis=1, q=0.25), np.quantile(LC[:,1,:], axis=1, q=0.75), color=color[1], alpha=0.3, edgecolor=None)
        ay.plot(range(1,201), np.mean(LC[:,2,:], axis=1), lw=1.5, color=color[2], linestyle='dashdot')
        ay.fill_between(range(1,201), np.quantile(LC[:,2,:], axis=1, q=0.25), np.quantile(LC[:,2,:], axis=1, q=0.75), color=color[2], alpha=0.3, edgecolor=None)

        ax.set_ylim(0,1)
        BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
        ax.plot(1+np.ones(10)*np.argmin(np.mean(LC[:,1,:], axis=1)), np.linspace(BOT,TOP,10), color="k", lw=1., alpha=0.5, zorder=3)

        ax.set_ylim(BOT,TOP); ax.set_xlabel('epoch'); ax.set_ylabel(r'MCR'); ay.set_ylabel(r'regularizer'); plt.tight_layout()
        plt.savefig("./Plots/plot5/%s_%s_dim-%d_size-%d_CMCR.png"%(METHs[me].split("/")[0],TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

