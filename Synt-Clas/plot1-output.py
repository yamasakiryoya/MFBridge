import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'dejavuserif'
plt.rcParams["font.size"] = 21
plt.rcParams['text.usetex'] = True
os.makedirs("./Plots/plot1-output", exist_ok=True)
color = ["black","red","green","blue","cyan","magenta","gold","purple","darkorange","gray","brown"]
marker = ["s","D","o","P","X","^","v","<",">","*","p"]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
args = parser.parse_args()
DATA = args.data

SEED  = 100
LAMs1 = [i/8.-6. for i in range(81)]
LAMs2 = [10.**(i/8.-6.) for i in range(81)]
SIZEs = [25,50,100,200,400]

def func(TYPEs,METHs,DIMs):
    for DIM,TYPE in itertools.product(DIMs,TYPEs):

        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.set_yscale('log'); ax.grid(True, which='both')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        # ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=[2.0, 5.0]))
        # ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10)))
        def label_only_2_5(x, pos):
            for v in [2, 5]:
                exponent = round(np.log10(x / v))
                if np.isclose(x, v * 10.0**exponent, rtol=1e-3): return f'{x:.1f}'
            return ''
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(label_only_2_5))
        def MyPlot(res,ME):
            for k in range(len(SIZEs)):
                ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1., color=color[ME])
            ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1., color=color[ME])
            ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker=marker[ME], lw=1., facecolor='None', edgecolors=color[ME], zorder=3)

        ME=0; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("Oracle",TYPE),delimiter=","); MyPlot(res,ME)

        ME=1; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%("ERM",TYPE),delimiter=","); MyPlot(res,ME)

        ME=2; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=3; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=4; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=5; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=6; res = np.loadtxt("Results/summary/%s-%s-CMCE.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel(r'$N$'); plt.ylabel('MCE for classification'); plt.tight_layout()
        plt.savefig("./Plots/plot1-output/%s_%d_CMCE.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        def MyPlot(res,ME):
            for k in range(len(SIZEs)):
                ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1., color=color[ME])
            ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1., color=color[ME])
            ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker=marker[ME], lw=1., facecolor='None', edgecolors=color[ME], zorder=3)

        ME=0; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("Oracle",TYPE),delimiter=","); MyPlot(res,ME)

        ME=1; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%("ERM",TYPE),delimiter=","); MyPlot(res,ME)

        ME=2; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=3; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=4; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=5; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=6; res = np.loadtxt("Results/summary/%s-%s-CMCR.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel(r'$N$'); plt.ylabel('MCR for classification'); plt.tight_layout()
        plt.savefig("./Plots/plot1-output/%s_%d_CMCR.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

        fig = plt.figure(figsize=(7.5,5)); ax = fig.add_subplot(111); ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        def MyPlot(res,ME):
            for k in range(len(SIZEs)):
                ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1., color=color[ME])
                plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1., color=color[ME])
            ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1., color=color[ME])
            ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker=marker[ME], lw=1., facecolor='None', edgecolors=color[ME], zorder=3)

        ME=1; res = np.loadtxt("Results/summary/%s-%s-CMFS.csv"%("ERM",TYPE),delimiter=","); MyPlot(res,ME)

        ME=2; res = np.loadtxt("Results/summary/%s-%s-CMFS.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=3; res = np.loadtxt("Results/summary/%s-%s-CMFS.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=4; res = np.loadtxt("Results/summary/%s-%s-CMFS.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=5; res = np.loadtxt("Results/summary/%s-%s-CMFS.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        ME=6; res = np.loadtxt("Results/summary/%s-%s-CMFS.csv"%(METHs[5*(ME-2)].split("-")[0],TYPE),delimiter=","); MyPlot(res,ME)

        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel(r'$N$'); plt.ylabel('MFS for classification'); plt.tight_layout()
        plt.savefig("./Plots/plot1-output/%s_%d_CMFS.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()

TYPEs = [DATA]
METHs = ["1MFBridge-0.1250000000/relu-4-20","1MFBridge-0.2500000000/relu-4-20","1MFBridge-0.5000000000/relu-4-20","1MFBridge-1.0000000000/relu-4-20","1MFBridge-2.0000000000/relu-4-20",
         "1MFElim-0.1250000000/relu-4-20","1MFElim-0.2500000000/relu-4-20","1MFElim-0.5000000000/relu-4-20","1MFElim-1.0000000000/relu-4-20","1MFElim-2.0000000000/relu-4-20",
         "1MFLoss-0.1250000000/relu-4-20","1MFLoss-0.2500000000/relu-4-20","1MFLoss-0.5000000000/relu-4-20","1MFLoss-1.0000000000/relu-4-20","1MFLoss-2.0000000000/relu-4-20",
         "1MFRisk-0.1250000000/relu-4-20","1MFRisk-0.2500000000/relu-4-20","1MFRisk-0.5000000000/relu-4-20","1MFRisk-1.0000000000/relu-4-20","1MFRisk-2.0000000000/relu-4-20",
         "1MFGrad-0.1250000000/relu-4-20","1MFGrad-0.2500000000/relu-4-20","1MFGrad-0.5000000000/relu-4-20","1MFGrad-1.0000000000/relu-4-20","1MFGrad-2.0000000000/relu-4-20"]
DIMs   = [10]
func(TYPEs,METHs,DIMs)
