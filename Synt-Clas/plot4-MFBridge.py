import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
os.makedirs("./Plots/plot4-MFBridge", exist_ok=True)
color = ["red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown"]
marker = ["D","o","P","X","^","v","<",">","*","p"]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--size', type=int)
args = parser.parse_args()
DATA = args.data
SIZE = args.size

SEED  = 100
LAMs1 = [i/8.-6. for i in range(81)]
LAMs2 = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [SIZE]

def func(TYPEs,DIMs):
    for DIM,TYPE,si in itertools.product(DIMs,TYPEs,range(len(SIZEs))):
        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.grid(True)
        fs = np.zeros((len(METHs)+1,DIM,SEED))

        k=0
        for se in range(SEED):
            tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%("1MFBridge-1.0000000000/relu-4-20",TYPE,DIM,SIZEs[si],0.,se), delimiter=",")
            fs[k,:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]
            if np.max(fs[k,:,se])!=0: fs[k,:,se] /= np.max(fs[k,:,se])
            else: fs[k,:,se] = 1.

        def MyFunc(METHs):
            res = np.zeros((len(METHs),len(LAMs2),SEED))
            ans = np.zeros((DIM,SEED))
            for me,la,se in itertools.product(range(len(METHs)),range(len(LAMs2)),range(SEED)):
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                res[me,la,se] = np.min(tmp[:,-4])
            for se in range(SEED):
                idx = np.unravel_index(np.argmin(res[:,:,se]), res[:,:,se].shape)
                tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[idx[0]],TYPE,DIM,SIZEs[si],LAMs2[idx[1]],se), delimiter=",")
                ans[:,se] = tmp[np.argmin(tmp[:,-4]),4:-4]
                if np.max(ans[:,se])!=0: ans[:,se] /= np.max(ans[:,se])
                else: ans[:,se] = 1.
            return ans

        k=1; fs[k] = MyFunc(METHs[k-1])

        k=2; fs[k] = MyFunc(METHs[k-1])

        k=3; fs[k] = MyFunc(METHs[k-1])

        k=4; fs[k] = MyFunc(METHs[k-1])

        k=5; fs[k] = MyFunc(METHs[k-1])

        k=6; fs[k] = MyFunc(METHs[k-1])

        def MyPlot(k):
            for l in range(DIM):
                ax.plot(l*np.ones(10), np.linspace(np.quantile(fs[k,l], q=0.25), np.quantile(fs[k,l], q=0.75), 10), lw=1., color=color[k])
                plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.25), lw=1., color=color[k])
                plt.plot(np.linspace(l-0.4,l+0.4,10), np.ones(10)*np.quantile(fs[k,l], q=0.50), lw=1., color=color[k])
                plt.plot(np.linspace(l-0.2,l+0.2,10), np.ones(10)*np.quantile(fs[k,l], q=0.75), lw=1., color=color[k])
            ax.scatter(range(DIM), np.mean(fs[k,:,:], axis=1), s=60, lw=1., facecolor='None', edgecolors=color[k], marker=marker[k], zorder=3)
        for k in range(len(METHs)+1): MyPlot(k)

        plt.xticks(np.arange(DIM),np.arange(1,DIM+1)); plt.xlabel(r'$j$'); plt.ylabel(r'$j$th RFS for classification, $N=%d$'%SIZEs[si], y=0.45); plt.tight_layout()
        plt.savefig("./Plots/plot4-MFBridge/%s_%d_%d_CRFSD.png"%(TYPE,DIM,SIZEs[si]), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

TYPEs = [DATA]
METHs = [["1MFBridge-0.1250000000/relu-4-20","1MFBridge-0.2500000000/relu-4-20","1MFBridge-0.5000000000/relu-4-20","1MFBridge-1.0000000000/relu-4-20","1MFBridge-2.0000000000/relu-4-20"],
         ["1MFBridge-0.1250000000/relu-4-20"],
         ["1MFBridge-0.2500000000/relu-4-20"],
         ["1MFBridge-0.5000000000/relu-4-20"],
         ["1MFBridge-1.0000000000/relu-4-20"],
         ["1MFBridge-2.0000000000/relu-4-20"]]
DIMs   = [10]
func(TYPEs,DIMs)
