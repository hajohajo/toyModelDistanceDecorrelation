import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

from dataGenerator import dataGenerator, trainingVariables
from neuralNetwork import createClassifier
from plotting import createMVADistribution, plotLossHistory, classifierVsX, createROCplot, initializePlotting, createJSDPlot, createShapeDistortionPlots, createVariableDistributions
from helperFunctions import getDiscoWeights

import pandas as pd
import numpy as np

def main():
    samples = int(1e6)
    testSamples = int(1e6)
    gen = dataGenerator()

    signal = gen.getSignal(nSamples=samples)
    background = gen.getBackground(nSamples=samples)
    dataset = pd.concat([signal, background], axis=0)
    dataset = dataset.sample(frac=1.0)

    listOfVariables = ["var1", "var2", "var3", "VOI"]
    binnings = [np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11), np.linspace(0.0, 10.0, 21), np.linspace(0.0, 700.0, 71)]

    weights = getDiscoWeights(dataset, "isSignal", "VOI", binningToUse=np.linspace(0.0, 700.0, 71))

    minValues = dataset.loc[:, trainingVariables].quantile(0.05).to_numpy()
    maxValues = dataset.loc[:, trainingVariables].quantile(0.95).to_numpy()

    classifier = createClassifier(len(trainingVariables), minValues, maxValues)

    targets = dataset.loc[:, "isSignal"]
    extraTargets = np.column_stack((targets, dataset.loc[:, "VOI"], weights))


    history = classifier.fit(dataset.loc[:, trainingVariables], extraTargets,
                   validation_split=0.1,
                   epochs=50,
                   batch_size=16384)

    testSignal = gen.getSignal(nSamples=testSamples*0.1)
    testBackground = gen.getBackground(nSamples=testSamples)
    testDataset = pd.concat([testSignal, testBackground], axis=0)
    testDataset = testDataset.sample(frac=1.0)

    prediction = classifier.predict(testDataset.loc[:, trainingVariables])

    testDataset.loc[:, "prediction"] = prediction

    initializePlotting()
    createVariableDistributions(dataset.loc[dataset.isSignal==1, :], listOfVariables, binnings, "Signal")
    createVariableDistributions(dataset.loc[dataset.isSignal!=1, :], listOfVariables, binnings, "Background")
    plotLossHistory(history)
    createMVADistribution(testDataset, name="toyAdversarial")
    createROCplot(testDataset, name="toyAdversarial")
    classifierVsX(testDataset, "VOI", np.linspace(0.0, 700.0, 71), plotName="toyAdversarial")
    createJSDPlot([testDataset.loc[testDataset.isSignal == 1], testDataset.loc[testDataset.isSignal != 1]], ["signal", "background"], np.linspace(0.0, 700.0, 71), "JSDplot")
    createShapeDistortionPlots(testDataset, "VOI", np.linspace(0.0, 700.0, 71), "distortion_")

    classifier.save("classifier.h5")

if __name__ == "__main__":
    main()