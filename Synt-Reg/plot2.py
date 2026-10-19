import os
import itertools
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
os.makedirs("./Plots/plot2", exist_ok=True)
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
SIZEs = [SIZE]#
DIMs   = [10]#
TYPEs = [DATA]#["non-additive-sq","quasi-multicolinear1","quasi-multicolinear2","quasi-multicolinear3","quasi-multicolinear4"]
METHs = [["1MFBridge-0.1250000000/relu-4-20","1MFBridge-0.2500000000/relu-4-20","1MFBridge-0.5000000000/relu-4-20","1MFBridge-1.0000000000/relu-4-20","1MFBridge-2.0000000000/relu-4-20"],
         ["1SPINN-0.0000000000/relu-4","1SPINN-0.2500000000/relu-4","1SPINN-0.5000000000/relu-4","1SPINN-0.7500000000/relu-4","1SPINN-1.0000000000/relu-4"],
         ["1LassoNet-0.1000000000/relu-4","1LassoNet-0.3162280000/relu-4","1LassoNet-1.0000000000/relu-4","1LassoNet-3.1622780000/relu-4","1LassoNet-10.0000000000/relu-4"],
         ["1STG-0.1250000000/relu-4","1STG-0.2500000000/relu-4","1STG-0.5000000000/relu-4","1STG-1.0000000000/relu-4","1STG-2.0000000000/relu-4"],
         ["1MFElim-0.1250000000/relu-4-20","1MFElim-0.2500000000/relu-4-20","1MFElim-0.5000000000/relu-4-20","1MFElim-1.0000000000/relu-4-20","1MFElim-2.0000000000/relu-4-20"],
         ["1MFLoss-0.1250000000/relu-4-20","1MFLoss-0.2500000000/relu-4-20","1MFLoss-0.5000000000/relu-4-20","1MFLoss-1.0000000000/relu-4-20","1MFLoss-2.0000000000/relu-4-20"],
         ["1MFRisk-0.1250000000/relu-4-20","1MFRisk-0.2500000000/relu-4-20","1MFRisk-0.5000000000/relu-4-20","1MFRisk-1.0000000000/relu-4-20","1MFRisk-2.0000000000/relu-4-20"],
         ["1Bridge-0.1250000000","1Bridge-0.2500000000","1Bridge-0.5000000000","1Bridge-1.0000000000","1Bridge-2.0000000000"],
         ["1SpAM-0.1250000000/relu-4","1SpAM-0.2500000000/relu-4","1SpAM-0.5000000000/relu-4","1SpAM-1.0000000000/relu-4","1SpAM-2.0000000000/relu-4"]][J]

#==================================================#
def func(TYPEs,METHs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        #
        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.set_yscale('log'); ax.grid(True, which='both')
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=[2, 5], numticks=10))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        #
        res1 = np.zeros((len(METHs),len(LAMs2),SEED))
        res2 = np.zeros((len(METHs),len(LAMs2),SEED))
        for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
            res1[me,la,se] = (tmp[np.argmin(tmp[:,-2]),-1])
            tmp2 = tmp[np.argmin(tmp[:,-2]),3:-2]
            if np.array_equal(np.array([0,1,2]),np.sort(np.argsort(tmp2)[::-1][:3]))==False: res2[me,la,se] += 1
        #
        def MyPlot(res1,k):
            ax.plot(np.arange(len(LAMs1)), np.mean(res1[k,:,:], axis=1), lw=1., color=color[k])
            # for l in range(len(LAMs1)):
            #     ax.plot(l*np.ones(10), np.linspace(np.quantile(res1[k,l], q=0.25), np.quantile(res1[k,l], q=0.75), 10), lw=.5, color=color[k])
            #     plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res1[k,l], q=0.25), lw=.5, color=color[k])
            #     plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res1[k,l], q=0.50), lw=.5, color=color[k])
            #     plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res1[k,l], q=0.75), lw=.5, color=color[k])
            # ax.scatter(np.arange(len(LAMs1)), np.mean(res1[k,:,:], axis=1), s=60, label=r"$q=2$", lw=.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        k = 0; MyPlot(res1,k)
        k = 1; MyPlot(res1,k)
        k = 2; MyPlot(res1,k)
        k = 3; MyPlot(res1,k)
        k = 4; MyPlot(res1,k)
        #
        # M = np.unravel_index(np.argmin(np.mean(res1, axis=2)),np.mean(res1, axis=2).shape)
        # L = (len(np.mean(res1[M[0],:,:], axis=1))-1)-np.argmin(np.mean(res1[M[0],:,:], axis=1)[::-1])
        # ax.scatter(npL, np.min(np.mean(res1[M[0],:,:], axis=1)), s=60, lw=1., facecolor='None', edgecolors="k", marker="o", zorder=3)
        # if npL<12: ax.text(npL+4, np.min(np.mean(res1[M[0],:,:], axis=1)), "%.4f"%np.min(np.mean(res1[M[0],:,:], axis=1)), va='center')
        # else: ax.text(npL-12, np.min(np.mean(res1[M[0],:,:], axis=1)), "%.4f"%np.min(np.mean(res1[M[0],:,:], axis=1)), va='center')
        #
        if METHs[0].split("-")[0]=="1MFBridge":
            abc = np.zeros((len(METHs),3,SEED))
            for me,se in itertools.product(range(len(METHs)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[3],TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
                abc[me,0,se] = tmp[np.argmin(tmp[:,-2]),0]#np.min(tmp[:,-2])
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
                abc[me,1,se] = tmp[np.argmin(tmp[:,-2]),1]
                tmp = np.loadtxt("./Results/1Const/%s/dim-%d/size-%d/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],se), delimiter=",")
                abc[me,2,se] = tmp[1]
            BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
            for me  in range(len(METHs)):
                serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
                plt.plot(np.ones(10)*8*(np.log10(serachlam)+6), np.linspace(BOT,TOP,10), color=color[me], lw=1., alpha=0.5, zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([8*i for i in range(11)],[-6,-5,-4,-3,-2,-1,0,1,2,3,4]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'MSE for regression, $N=%d$'%SIZEs[si], y=0.45); plt.tight_layout()#test risk#
        plt.savefig("./Plots/plot2/%s_%s_%d_%d_RMSE.png"%(METHs[0].split("-")[0],TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        #
        def MyPlot(res2,k):
            ax.plot(np.arange(len(LAMs1)), np.mean(res2[k,:,:], axis=1), lw=1., color=color[k])
            # for l in range(len(LAMs1)):
            #     ax.plot(l*np.ones(10), np.linspace(np.quantile(res2[k,l], q=0.25), np.quantile(res2[k,l], q=0.75), 10), lw=.5, color=color[k])
            #     plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res2[k,l], q=0.25), lw=.5, color=color[k])
            #     plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(res2[k,l], q=0.50), lw=.5, color=color[k])
            #     plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(res2[k,l], q=0.75), lw=.5, color=color[k])
            # ax.scatter(np.arange(len(LAMs1)), np.mean(res2[k,:,:], axis=1), s=60, label=r"$q=2$", lw=.5, facecolor='None', edgecolors=color[k], marker="v", zorder=3)
        k = 0; MyPlot(res2,k)
        k = 1; MyPlot(res2,k)
        k = 2; MyPlot(res2,k)
        k = 3; MyPlot(res2,k)
        k = 4; MyPlot(res2,k)
        #
        # M = np.unravel_index(np.argmin(np.mean(res2, axis=2)),np.mean(res2, axis=2).shape)
        # L = (len(np.mean(res2[M[0],:,:], axis=1))-1)-np.argmin(np.mean(res2[M[0],:,:], axis=1)[::-1])
        # ax.scatter(L, np.min(np.mean(res2[M[0],:,:], axis=1)), s=60, lw=1., facecolor='None', edgecolors="k", marker="o", zorder=3)
        # if L<12: ax.text(L+4, np.min(np.mean(res2[M[0],:,:], axis=1)), "%.4f"%np.min(np.mean(res2[M[0],:,:], axis=1)), va='center')
        # else: ax.text(L-12, np.min(np.mean(res2[M[0],:,:], axis=1)), "%.4f"%np.min(np.mean(res2[M[0],:,:], axis=1)), va='center')
        #
        if METHs[0].split("-")[0]=="1MFBridge":
            abc = np.zeros((len(METHs),3,SEED))
            for me,se in itertools.product(range(len(METHs)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[3],TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
                abc[me,0,se] = tmp[np.argmin(tmp[:,-2]),0]#np.min(tmp[:,-2])
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[0],se), delimiter=",")
                abc[me,1,se] = tmp[np.argmin(tmp[:,-2]),1]
                tmp = np.loadtxt("./Results/1Const/%s/dim-%d/size-%d/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],se), delimiter=",")
                abc[me,2,se] = tmp[1]
            BOT = ax.get_ylim()[0]; TOP = ax.get_ylim()[1]
            for me  in range(len(METHs)):
                serachlam = (np.mean(abc[me,2,:])-np.mean(abc[me,0,:]))/np.mean(abc[me,1,:])
                plt.plot(np.ones(10)*8*(np.log10(serachlam)+6), np.linspace(BOT,TOP,10), color=color[me], lw=1., alpha=0.5, zorder=1)
        #
        plt.ylim([BOT,TOP]); plt.xticks([8*i for i in range(11)],[-6,-5,-4,-3,-2,-1,0,1,2,3,4]); plt.xlabel(r'$\log_{10}(\lambda)$'); plt.ylabel(r'MFS for regression, $N=%d$'%SIZEs[si], y=0.45); plt.tight_layout()#test risk#
        plt.savefig("./Plots/plot2/%s_%s_%d_%d_RMFS.png"%(METHs[0].split("-")[0],TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

func(TYPEs,METHs,DIMs)
# METHs = ["1MFSCAD-0.0625000000/relu-4-10","1MFSCAD-0.1250000000/relu-4-10","1MFSCAD-0.2500000000/relu-4-10","1MFSCAD-0.5000000000/relu-4-10","1MFSCAD-1.0000000000/relu-4-10","1MFSCAD-2.0000000000/relu-4-10"]
# func(TYPEs,METHs,DIMs)
# METHs = ["1MFMCP-0.0625000000/relu-4-10","1MFMCP-0.1250000000/relu-4-10","1MFMCP-0.2500000000/relu-4-10","1MFMCP-0.5000000000/relu-4-10","1MFMCP-1.0000000000/relu-4-10","1MFMCP-2.0000000000/relu-4-10"]
# func(TYPEs,METHs,DIMs)
# METHs = ["1MFLogSum-0.0000100000/relu-4-10","1MFLogSum-0.0001000000/relu-4-10","1MFLogSum-0.0010000000/relu-4-10","1MFLogSum-0.0100000000/relu-4-10","1MFLogSum-0.1000000000/relu-4-10","1MFLogSum-1.0000000000/relu-4-10"]
# func(TYPEs,METHs,DIMs)
# METHs = ["1MFAdaLasso-0.5000000000/relu-4-10","1MFAdaLasso-1.0000000000/relu-4-10","1MFAdaLasso-2.0000000000/relu-4-10","1MFAdaLasso-4.0000000000/relu-4-10","1MFAdaLasso-8.0000000000/relu-4-10","1MFAdaLasso-16.0000000000/relu-4-10"]
# func(TYPEs,METHs,DIMs)
# METHs = ["1MFAdaBLasso-0.5000000000/relu-4-10","1MFAdaBLasso-1.0000000000/relu-4-10","1MFAdaBLasso-2.0000000000/relu-4-10","1MFAdaBLasso-4.0000000000/relu-4-10","1MFAdaBLasso-8.0000000000/relu-4-10","1MFAdaBLasso-16.0000000000/relu-4-10"]
# func(TYPEs,METHs,DIMs)