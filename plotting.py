import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from scipy.stats import norm
from scipy.spatial.distance import jensenshannon

sns.set()
sns.set_style("whitegrid")

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams
from matplotlib import font_manager
font_manager._rebuild()
rcParams['font.serif'] = "Latin Modern Sans"
rcParams['font.family'] = "serif"

pathToPlots = "./"

def initializePlotting():
    listOfFolders = [pathToPlots+"plots", pathToPlots+"plots/distortionPlots"]
    for dir in listOfFolders:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

#Production worthy
def classifierVsX(dataframe, variable, binningToUse, plotName):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 6))

    fig.suptitle("Mean output with respect to variable of interest")
    variableName = variable
    binning = binningToUse
    binCenters = binning+(binning[1]-binning[0])/2.0
    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]
    labels = ["background", "signal"]
    for massPoint in np.unique(dataframe.loc[:, "massPoint"]):
        isSignal = 0
        if massPoint != 1.0:
            isSignal = 1
        mva = dataframe.loc[(dataframe.massPoint == massPoint), "prediction"]
        variable = dataframe.loc[(dataframe.massPoint == massPoint), variableName]
        indices = np.digitize(variable, binning, right=True)
        indices[indices == 0] = 1
        indices = indices - 1

        binMean = -np.ones(len(binning))
        binStd = np.zeros(len(binning))

        for i in np.unique(indices):
            if(np.sum(indices == i)<2):
                continue
            mean, std = norm.fit(mva[indices == i])

            binMean[i] = mean
            binStd[i] = std

        up = np.add(binMean, binStd)
        down = np.add(binMean, -binStd)

        nonzeroPoints = (binMean > -1)
        if(massPoint==250.0 or massPoint==1):
            ax0.fill_between(binCenters[nonzeroPoints], up[nonzeroPoints], down[nonzeroPoints], alpha=0.6, color=colors[isSignal]) #label="$\pm 1\sigma$",
            ax0.plot(binCenters[nonzeroPoints], up[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.8)
            ax0.plot(binCenters[nonzeroPoints], down[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.8)
            ax0.scatter(binCenters[nonzeroPoints], binMean[nonzeroPoints], c=colors[isSignal], label=labels[isSignal], linewidths=1.0, edgecolors='k')
        else:
            ax0.fill_between(binCenters[nonzeroPoints], up[nonzeroPoints], down[nonzeroPoints],
                             alpha=0.6, color=colors[isSignal])
            ax0.plot(binCenters[nonzeroPoints], up[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.7)
            ax0.plot(binCenters[nonzeroPoints], down[nonzeroPoints], c=colors[isSignal], linewidth=1.0, alpha=0.7)
            ax0.scatter(binCenters[nonzeroPoints], binMean[nonzeroPoints], c=colors[isSignal],
                        linewidths=1.0, edgecolors='k')


    ax0.set_ylim(-0.05, 1.35)
    locs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax0.set_yticks(locs)  # Set locations and labels
    ax0.set_xlim(binCenters[0], binCenters[-1])
    ax0.set_ylabel("Mean DNN output")

    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[::-1], labels[::-1], loc='upper right')

    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.grid(False, axis='x')

    sig = dataframe.loc[(dataframe.isSignal == 1), "VOI"]
    bkg = dataframe.loc[(dataframe.isSignal != 1), "VOI"]
    ax1.hist((bkg, sig), bins=binning, label=labels, color=colors, stacked=True, edgecolor="black", linewidth=1.0, alpha=0.7)

    ax1.set_xlabel("Variable of interest")
    ax1.set_ylabel("Number of events/bin")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(False, axis='x')

    plt.tight_layout(pad=2.2)
    fig.align_ylabels((ax0, ax1))
    plt.savefig(pathToPlots+"plots/"+plotName+".pdf")
    plt.clf()


def createROCplot(dataframe, name):
    falsePositiveRate, truePositiveRate, thresholds = roc_curve(dataframe.loc[:, "isSignal"], dataframe.loc[:, "prediction"])

    _auc = auc(falsePositiveRate, truePositiveRate)

    plt.plot(truePositiveRate, 1-falsePositiveRate, label="ROC area = {:.3f}".format(_auc))
    plt.legend()
    plt.title("ROC curves")
    plt.ylabel("Fake rejection")
    plt.xlabel("True efficiency")
    plt.savefig(pathToPlots+"plots/"+name+"_ROC.pdf")
    plt.clf()

def createMVADistribution(dataframe, name):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("MVA distributions for signal and background")

    binning = np.linspace(0.0, 1.0, 41)

    colors = [sns.xkcd_rgb["rouge"], sns.xkcd_rgb["cerulean"]]
    isSignal = dataframe.loc[:, "isSignal"]
    plt.hist(dataframe.loc[(isSignal == 1), "prediction"], bins=binning, label="Signal", density=True, alpha=0.7, edgecolor="black", linewidth=1.0, color=colors[0])
    plt.hist(dataframe.loc[(isSignal == 0), "prediction"], bins=binning, label="Background", density=True, alpha=0.7, edgecolor="black", linewidth=1.0, color=colors[1])

    plt.legend()
    plt.xlabel("MVA output")
    plt.ylabel("Events")

    plt.savefig(pathToPlots+"plots/" + name + "_MVADistribution.pdf")
    plt.clf()

def plotLossHistory(history):
    colors = [sns.xkcd_rgb["gold"], sns.xkcd_rgb["grass"]]

    loss = history.history["loss"][1:]
    valLoss = history.history["val_loss"][1:]
    epochs = range(1, len(loss)+2)[1:]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label="Training loss", color=colors[0], linewidth=2.0)
    plt.plot(epochs, valLoss, label="Validation loss", color=colors[1], linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss function value")
    plt.yscale('log')
    fig.suptitle("Training and validation loss")
    plt.legend()
    plt.savefig(pathToPlots+"plots/Losses.pdf")
    plt.clf()

def createJSDPlot(listOfDatasets, listOfNames, binningToUse, name):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("JSD scores for different cut thresholds")

    colors = [sns.xkcd_rgb["rouge"], sns.xkcd_rgb["cerulean"]]

    binning = binningToUse
    grid = np.linspace(0.0, 1.0, 21)

    for ind in range(len(listOfDatasets)):
        dataset = listOfDatasets[ind]

        #Skip datasets with no entries or too small statistic
        if(dataset.empty or dataset.shape[0]<100):
            continue

        variable = dataset.loc[:, "VOI"].to_numpy()
        predictions = dataset.loc[:, "prediction"].to_numpy()

        jseValues = []
        binnedAll, _ = np.histogram(variable, binning)
        for threshold in grid:
            passes = (predictions > threshold)
            passingBinned, _ = np.histogram(variable[passes], binning)

            #If no entries left after selection, maximal distortion
            if(np.sum(passingBinned)==0.0):
                jseValues.append((1.0))
                continue

            jse = jensenshannon(passingBinned, binnedAll)
            jseValues.append(jse)

        plt.scatter(grid, jseValues, s=120, marker='.', alpha=0.7, linewidths=1.0, label=listOfNames[ind]+', '+str(dataset.shape[0]), edgecolors='k', color=colors[ind])

    plt.xlabel("Cut value")
    plt.ylabel("JSD Score")
    plt.legend()
    plt.savefig("plots/"+name+'.pdf')
    plt.clf()

def createShapeDistortionPlots(dataframe, variableName, binningToUse, name):
    binning = binningToUse
    binWidth = binning[1]-binning[0]
    binCenters = binning+binWidth/2.0
    binCenters = binCenters[:-1]
    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]
    labels = ["before cut bkg", "before cut sig+bkg"]
    selectedLabels = ["after cut bkg", "after cut sig+bkg"]


    sigIndices = dataframe.isSignal == 1
    bkgIndices = dataframe.isSignal != 1
    sig = dataframe.loc[sigIndices, variableName]
    bkg = dataframe.loc[bkgIndices, variableName]

    workingPoints = np.linspace(0.0, 1.0, 21)

    for point in workingPoints:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 1]},
                                       figsize=(8, 6))
        fig.suptitle("Distortion after cut at {:.2f}".format(point))
        selectedSig = sig.loc[dataframe.loc[sigIndices, "prediction"] >= point]
        selectedBkg = bkg.loc[dataframe.loc[bkgIndices, "prediction"] >= point]

        bins, _, _ = ax0.hist((bkg, sig), bins=binning, label=labels, color=colors, stacked=True, histtype="step", linewidth=1.5)
        selectedBins, _, _ = ax0.hist((selectedBkg, selectedSig), bins=binning, label=selectedLabels, color=colors, stacked=True, histtype="step", linestyle='dashed', linewidth=1.5)
        ax0.legend(loc='upper right')

        bkgOnlyBins = bins[0]
        bins = bins[1]
        bkgOnlySelectedBins = selectedBins[0]
        selectedBins = selectedBins[1]

        # ax0.set_xlabel("Variable of interest")
        ax0.set_ylabel("Number of events/bin")
        ax0.set_ylim(0.0, np.max(bins)+np.max(bins)/1.5)

        #Ratio plot
        ratio = np.divide(selectedBins, bins)
        bkgOnlyRatio = np.divide(bkgOnlySelectedBins, bkgOnlyBins)
        ax1.plot([binning[0], binning[-1]], [1, 1], linestyle='--', linewidth=1.5, color='k')
        ax1.plot(np.insert(np.append(binCenters, binCenters[-1]+binWidth), 0, binCenters[0]-binWidth), np.insert(np.append(ratio, ratio[-1]), 0, ratio[0]), linewidth=1.5, color='k', label="sig+bkg")
        ax1.plot(binCenters, bkgOnlyRatio, linewidth=1.5, linestyle=':', color='k', label="bkg")
        ax1.set_xlabel(variableName)
        ax1.set_xlim(binning[0], binning[-1])
        ax1.set_ylabel("Ratio")
        ax1.legend()
        plt.savefig("plots/distortionPlots/"+name+"cutValue_{:.2f}.pdf".format(point))
        plt.clf()

def createVariableDistributions(dataframe, listOfVariables, listOfBinnings, plotName):
    ncols = int(len(listOfVariables)/2.0)
    nrows = int(np.ceil(len(listOfVariables)/ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    fig.suptitle("Distributions for samples")
    axs = axs.flatten()

    binnings = listOfBinnings
    variableNames = listOfVariables

    colors = [sns.xkcd_rgb["cerulean"], sns.xkcd_rgb["rouge"]]
    colorToUse = colors[dataframe.loc[0, "isSignal"]==1]

    for i in range(len(variableNames)):
        name = variableNames[i]
        variable = dataframe.loc[:, name]
        binning = binnings[i]
        ax = axs[i]

        ax.hist(variable, bins=binning, linewidth=1.0, edgecolor='k', color=colorToUse, alpha=0.7)
        ax.set_title(name)
        ax.set_xlim(binning[0], binning[-1])

    plt.tight_layout(pad=3.5)
    plt.savefig("plots/VariableDistributions_"+plotName+".pdf")

