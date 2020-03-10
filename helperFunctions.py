import numpy as np

#getDiscoWeights provides weights that are suitably flattened for distance decorrelating
#loss function. Assumes inputs
#
# dataframe:            Containing at least columns signalVariable and variableToFlatten
# signalVariable:       Name of the column with values 1 for signal and something else for background
# variableToFlatten:    Name of the column with variable of interest we want to decorrelate from (i.e. "TransverseMass")
# binningToUse:         Bins to flatten the background into. Should choose binning so that there are enough
#                       background events in each bin, but there is no explicit test checking for that.
def getDiscoWeights(dataframe, signalVariable, variableToFlatten, binningToUse):
    variable = dataframe.loc[:, variableToFlatten]
    nEntries = dataframe.shape[0]
    binning = binningToUse
    signalIndices = dataframe.loc[:, signalVariable] == 1
    backgroundIndices = np.logical_not(signalIndices)

    variable = np.clip(variable, binning[0] + 1e-3, binning[-1] - 1e-3)

    digitizedIndices = np.digitize(variable[backgroundIndices], binning, right=True)

    #Fix the discrepancy since indexing and digitize are not compatible as such
    digitizedIndices[digitizedIndices == 0] = 1
    digitizedIndices = digitizedIndices - 1

    uniques = np.unique(digitizedIndices)

    binMultipliers = np.zeros(len(uniques))
    targetBinContent = nEntries/len(uniques)

    for i in np.unique(digitizedIndices):
        inBin = np.sum(digitizedIndices == i)
        if(inBin > 0):
            binMultipliers[i] = targetBinContent/inBin

    #Give each background event its weight, leaves signal event weights to zeros
    sampleWeights = np.zeros(nEntries)
    sampleWeights[backgroundIndices] = binMultipliers[digitizedIndices]

    # import matplotlib.pyplot as plt
    # plt.hist(variable[signalIndices], bins=binning, weights=sampleWeights[signalIndices], label="true", alpha=0.8)
    # plt.hist(variable[backgroundIndices], bins=binning, weights=sampleWeights[backgroundIndices], label="fake", alpha=0.8)
    # plt.legend()
    # # plt.xscale("log")
    # plt.show()
    return sampleWeights
