import numpy as np
import pandas as pd

#variables for training
trainingVariables = ["var1", "var2", "var3", "VOI"]

#Class for generating toy data to demonstrate classifier decorrelation
class dataGenerator:
    def __init__(self):
        self.variableNames = ["var1", "var2", "var3", "VOI", "isSignal"]
        self.ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 300.0), (0.0, 300.0)]

    def getSignal(self, nSamples):
        # massPoints = [100, 250, 400, 550]
        massPoints = [250]

        width = 20
        df = pd.DataFrame(columns=self.variableNames+["massPoint"])
        for point in massPoints:
            _nSamples = int(nSamples/len(massPoints))
            var1 = np.arcsin(np.random.uniform(self.ranges[0][0], self.ranges[0][1], _nSamples))
            var2 = np.random.uniform(self.ranges[1][0], self.ranges[1][1], _nSamples)
            VOI = np.random.normal(point, width, _nSamples)
            var3 = np.log(1.0+VOI.copy())
            massPoint = np.ones(_nSamples)*point
            isSignal = np.ones(_nSamples)
            dataframe = pd.DataFrame(data=np.column_stack((var1, var2, var3, VOI, isSignal, massPoint)), columns=self.variableNames+["massPoint"])
            df = pd.concat([df, dataframe], axis=0)
        return df


    def getBackground(self, nSamples):
        var1 = np.arccos(np.random.uniform(self.ranges[0][0], self.ranges[0][1], nSamples))
        var2 = np.random.uniform(self.ranges[1][0], self.ranges[1][1], nSamples)
        VOI = np.random.exponential(300, nSamples)
        # VOI = np.random.uniform(0.0, 800.0, nSamples)
        # var3 = np.log(1.0+VOI)
        var3 = np.log(1.0+VOI.copy())
        isSignal = np.zeros(nSamples)
        massPoint = np.ones(nSamples)

        dataframe = pd.DataFrame(data=np.column_stack((var1, var2, var3, VOI, isSignal, massPoint)), columns=self.variableNames+["massPoint"])
        return dataframe