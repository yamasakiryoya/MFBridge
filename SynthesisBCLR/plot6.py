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
if os.path.isdir("./Plots/plot6")==False: os.makedirs("./Plots/plot6")
color = ["red","green","blue","cyan","magenta","gold","black"]#["gold","magenta","orange","saddlebrown","gray"]#

SEED  = 20
LAMs1 = [i/4.-5. for i in range(33)]
LAMs2 = [10.**(i/4.-5.) for i in range(33)]
SIZEs = [25,50,100,200,400]

def func(TYPEs,METHs,DIMs):
    for DIM,TYPE in itertools.product(DIMs,TYPEs):
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
            print("./Results/NGLasso1-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se))
            res[si,se] = tmp[np.argmin(tmp[:,-4]),-2]
        #
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
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[1])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[1])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFBridge", lw=1.5, facecolor='None', edgecolors=color[1], zorder=3)
        # SCAD
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5,10):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[2])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[2])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFSCAD", lw=1.5, facecolor='None', edgecolors=color[2], zorder=3)
        # MCP
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(10,15):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[3])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[3])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFMCP", lw=1.5, facecolor='None', edgecolors=color[3], zorder=3)
        # LogSum
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(15,20):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[4])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[4])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[4])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[4])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[4])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFLogSum", lw=1.5, facecolor='None', edgecolors=color[4], zorder=3)
        # AdaLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(20,25):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-4]), tmp[np.argmin(tmp[:,-4]),-2]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[5])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[5])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[5])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[5])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[5])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFAdaLasso", lw=1.5, facecolor='None', edgecolors=color[5], zorder=3)
        #
        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel('n'); plt.ylabel('MCEE'); plt.tight_layout()
        plt.savefig("./Plots/plot6/%s_dim-%d_Closs.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
        #==================================================#
        fig = plt.figure(figsize=(9,5)); cmap = plt.get_cmap('jet'); ax = fig.add_subplot(111); ax.grid(True)
        # ERM
        res = np.zeros((len(SIZEs),SEED))
        for si,se in itertools.product(range(len(SIZEs)),range(SEED)):
            tmp = np.loadtxt("./Results/NGLasso1-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se), delimiter=",")
            print("./Results/NGLasso1-0.0000000000/relu-4/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(TYPE,DIM,SIZEs[si],0.0,se))
            res[si,se] = tmp[np.argmin(tmp[:,-3]),-1]
        #
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
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[1])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[1])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[1])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFBridge", lw=1.5, facecolor='None', edgecolors=color[1], zorder=3)
        # SCAD
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(5,10):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[2])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[2])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[2])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFSCAD", lw=1.5, facecolor='None', edgecolors=color[2], zorder=3)
        # MCP
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(10,15):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[3])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[3])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[3])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFMCP", lw=1.5, facecolor='None', edgecolors=color[3], zorder=3)
        # LogSum
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(15,20):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[4])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[4])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[4])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[4])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[4])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFLogSum", lw=1.5, facecolor='None', edgecolors=color[4], zorder=3)
        # AdaLasso
        res = np.zeros((len(SIZEs),SEED))
        for si in range(len(SIZEs)):
            tmp1, tmp2, tmp3 = np.zeros(SEED), np.zeros(SEED), 10.**8*np.ones(SEED)
            for me in range(20,25):
                result = np.zeros((len(LAMs2),20))
                for se,la in itertools.product(range(SEED),range(len(LAMs2))):
                    tmp = np.loadtxt("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se), delimiter=",")
                    print("./Results/%s/%s/dim-%d/size-%d/lam-%.10f/1-seed-%d.csv"%(METHs[me],TYPE,DIM,SIZEs[si],LAMs2[la],se))
                    tmp1[se], tmp2[se] = np.min(tmp[:,-3]), tmp[np.argmin(tmp[:,-3]),-1]
                    if tmp1[se]<tmp3[se]: tmp3[se] = tmp1[se]; res[si,se] = tmp2[se]
        #
        for k in range(len(SIZEs)):
            ax.plot(k*np.ones(10), np.linspace(np.quantile(res[k], q=0.25), np.quantile(res[k], q=0.75), 10), lw=1.5, color=color[5])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.25), lw=1.5, color=color[5])
            plt.plot(np.linspace(k-0.4,k+0.4,10), np.ones(10)*np.quantile(res[k], q=0.50), lw=1.5, color=color[5])
            plt.plot(np.linspace(k-0.2,k+0.2,10), np.ones(10)*np.quantile(res[k], q=0.75), lw=1.5, color=color[5])
        ax.plot(range(len(SIZEs)), np.mean(res, axis=1), lw=1.5, color=color[5])
        ax.scatter(range(len(SIZEs)), np.mean(res, axis=1), s=100, marker="o", label="MFAdaLasso", lw=1.5, facecolor='None', edgecolors=color[5], zorder=3)
        #
        plt.xticks(range(len(SIZEs)),SIZEs); plt.xlabel('n'); plt.ylabel('MCR'); plt.tight_layout()
        plt.savefig("./Plots/plot6/%s_dim-%d_miscla.png"%(TYPE,DIM), bbox_inches="tight", pad_inches=.02, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none'); plt.close()
#
#"""
TYPEs = ["non-additive-sq"]
METHs = ["NBridge1-0.1250000000/relu-4-10","NBridge1-0.2500000000/relu-4-10","NBridge1-0.5000000000/relu-4-10","NBridge1-1.0000000000/relu-4-10","NBridge1-2.0000000000/relu-4-10",
         "NSCAD1-1.0000000000/relu-4-10","NSCAD1-2.0000000000/relu-4-10","NSCAD1-3.0000000000/relu-4-10","NSCAD1-4.0000000000/relu-4-10","NSCAD1-5.0000000000/relu-4-10",
         "NMCP1-1.0000000000/relu-4-10","NMCP1-2.0000000000/relu-4-10","NMCP1-3.0000000000/relu-4-10","NMCP1-4.0000000000/relu-4-10","NMCP1-5.0000000000/relu-4-10",
         "NLogSum1-0.1000000000/relu-4-10","NLogSum1-0.0100000000/relu-4-10","NLogSum1-0.0010000000/relu-4-10","NLogSum1-0.0001000000/relu-4-10","NLogSum1-0.0000100000/relu-4-10",
         "NAdaLasso1-0.2500000000/relu-4-10","NAdaLasso1-0.5000000000/relu-4-10","NAdaLasso1-1.0000000000/relu-4-10","NAdaLasso1-2.0000000000/relu-4-10","NAdaLasso1-2.0000000000/relu-4-10"]
DIMs   = [10]
func(TYPEs,METHs,DIMs)