import itertools
import os

import numpy as np
import pandas as pd
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

PC_VALUE_AT_RADIUS = 0.6  # 0.2 activation value (constant K in pc activation formula)

TRAIN = 0
VALIDATION = 1
ALL = 2


class DataGenerator(keras.utils.Sequence):
    def __init__(self, freedom=2, mode=TRAIN):
        if mode == TRAIN:
            self.FAST = [3, 4, 6, 7]
            self.MEDIUM = [8, 9, 11, 12]
            self.SLOW = [14, 15, 17, 18]
        elif mode == VALIDATION:
            self.FAST = [2, 5]
            self.MEDIUM = [10, 13]
            self.SLOW = [16, 19]
        elif mode == ALL:
            self.FAST = [2, 3, 4, 5, 6, 7]
            self.MEDIUM = [8, 9, 10, 11, 12, 13]
            self.SLOW = [14, 15, 16, 17, 18, 19]
        else:
            self.FAST = []
            self.MEDIUM = []
            self.SLOW = []

        self.train = (mode == TRAIN)

        self.freedom = freedom

        self.paths = []

        self.__create_paths__()

        self.on_epoch_end()

    def __createSequencesInPath__(self, sequence):
        MAX = 0.9
        MIN = -0.9
        numberOfSegments = len(sequence)
        segmentsSize = (MAX - MIN) / numberOfSegments

        start = [MIN, MIN]
        end = [MIN + segmentsSize, MIN + segmentsSize]

        segmentList = []

        for segmentSteps in sequence[0]:
            for steps in segmentSteps:
                segment = np.linspace(start=start, stop=end, num=steps, endpoint=False)
                segmentList.append(segment)
                start = end
                end = [end[0] + segmentsSize, end[1] + segmentsSize]

            segmentList.append([[MAX, MAX]])
            path = np.concatenate(segmentList)

            self.paths.append((pd.DataFrame(path, columns=['x', 'y']), sequence[1]))

    def __calc_activation_matrix__(self, path, pcs, localization_noise=0, percentual_noise=0, additive_noise=0):
        """ Calculate a matrix containing the activation of all place cells for all times.
            Each row represents a place cell, while columns represent the time index.
            Both 'pos' and 'pcs' are data frames containing the path and the set of place cells.
        """
        # get number of pcs and position in path
        num_pcs = len(pcs)
        num_pos = len(path)

        # convert data to numpy to operate
        radii = pcs['placeradius'].to_numpy()
        pcs = pcs[['x', 'y']].to_numpy()
        pos = path[['x', 'y']].to_numpy()

        # replicate the position vector by the number of place cells for easy operations
        pos_tile = pos.reshape(1, -1, 2)
        pos_all = np.tile(pos_tile, (num_pcs, 1, 1))

        # replicate the place cells and radii by the number of positions for easy operations
        pcs_tile = pcs.reshape(-1, 1, 2)
        pcs_all = np.tile(pcs_tile, (1, num_pos, 1))
        radii_all = np.tile(radii.reshape((-1, 1)), (1, num_pos))

        # calculate the activations (see description of formula at the top of this file)
        delta = pos_all - pcs_all
        delta2 = (delta * delta).sum(2)

        if localization_noise != 0:
            d_error = np.random.uniform(1 - localization_noise, 1 + localization_noise, delta2.shape)
            delta2 *= d_error * d_error

        r2 = radii_all * radii_all
        exponents = np.log(PC_VALUE_AT_RADIUS) * delta2 / r2
        activations = np.exp(exponents)

        if percentual_noise != 0:
            activations *= np.random.uniform(1 - percentual_noise, 1 + percentual_noise, activations.shape)

        if additive_noise != 0:
            activations = np.minimum(
                np.maximum(0, activations + np.random.uniform(-additive_noise, additive_noise, activations.shape)), 1)
        return activations

    def __getYData__(self, dataValue):
        compareData = []

        if self.freedom == 2:
            compareData = [90, 270]
        elif self.freedom == 8:
            compareData = [0, 45, 90, 135, 180, 225, 270, 315]

        y = [0] * len(compareData)

        for i in range(len(compareData)):
            if dataValue == compareData[i]:
                y[i] = 1
                break

        return y

    def generate_sequences(self, rawDataList, numberOfSegments, segmentsSize, pcs):
        start_values, end_values = self.__initialize_segments__(segmentsSize, numberOfSegments)

        self.__generate_sequence_recursive__(rawDataList, numberOfSegments, start_values, end_values, [], pcs)

    def __initialize_segments__(self, segmentsSize, numberOfSegments):
        MAX = 0.9
        MIN = -0.9

        start_values = [[MIN, MIN]]
        end_values = [[MIN + segmentsSize, MIN + segmentsSize]]

        for _ in range(1, numberOfSegments):
            start = end_values[-1]
            end = [start[0] + segmentsSize, start[1] + segmentsSize]
            start_values.append(start)
            end_values.append(end)

        return start_values, end_values

    def __generate_sequence_recursive__(self, rawDataList, numberOfSegments, start_values, end_values, sequence, pcs):
        if numberOfSegments == 0:
            # Convert the sequence into a path DataFrame
            path = pd.DataFrame(np.concatenate(sequence), columns=['x', 'y'])

            # Get the output value based on the current outputType
            outputType = len(rawDataList) - numberOfSegments
            output = self.__getYData__(rawDataList[outputType][1])

            # Calculate activation matrix
            activation = self.__calc_activation_matrix__(path, pcs)

            # Append activation and output to the dataList
            self.dataList.append((activation, output))
            return

        for i in range(len(self.SLOW)):
            segmentTypesList = rawDataList[len(rawDataList) - numberOfSegments][0]
            steps = segmentTypesList[len(segmentTypesList) - numberOfSegments][i]
            segment = np.linspace(start=start_values[len(segmentTypesList) - numberOfSegments],
                                  stop=end_values[len(segmentTypesList) - numberOfSegments], num=steps, endpoint=False)

            # Append the segment to the sequence
            sequence.append(segment)

            # Recursive call with decreased numberOfSegments
            self.__generate_sequence_recursive__(rawDataList, numberOfSegments - 1, start_values, end_values, sequence,
                                                 pcs)

            # Remove the last segment to backtrack
            sequence.pop()

    def __create_paths__(self):
        rawDataList = []
        if self.freedom == 2:
            DEGREE_90 = [self.SLOW, self.FAST, self.SLOW]
            DEGREE_270 = [self.SLOW, self.SLOW, self.SLOW]
        elif self.freedom == 8:
            DEGREE_0 = [self.SLOW, self.SLOW, self.SLOW, self.SLOW]
            DEGREE_45 = [self.FAST, self.SLOW, self.FAST, self.SLOW]
            DEGREE_90 = [self.MEDIUM, self.FAST, self.SLOW, self.SLOW]
            DEGREE_135 = [self.SLOW, self.SLOW, self.MEDIUM, self.SLOW]
            DEGREE_180 = [self.SLOW, self.MEDIUM, self.SLOW, self.SLOW]
            DEGREE_225 = [self.SLOW, self.MEDIUM, self.FAST, self.SLOW]
            DEGREE_270 = [self.SLOW, self.FAST, self.MEDIUM, self.SLOW]
            DEGREE_315 = [self.SLOW, self.FAST, self.FAST, self.SLOW]
            rawDataList.append((DEGREE_0, 0))
            rawDataList.append((DEGREE_45, 45))
            rawDataList.append((DEGREE_135, 135))
            rawDataList.append((DEGREE_180, 180))
            rawDataList.append((DEGREE_225, 225))
            rawDataList.append((DEGREE_315, 315))

        rawDataList.append((DEGREE_90, 90))
        rawDataList.append((DEGREE_270, 270))

        MAX = 0.9
        MIN = -0.9

        self.dataList = []

        numberOfSegments = len(rawDataList[0][0])
        segmentsSize = (MAX - MIN) / numberOfSegments

        pcs = pd.read_csv('input/placecells.csv', delim_whitespace=True)

        start1 = [MIN, MIN]
        end1 = [MIN + segmentsSize, MIN + segmentsSize]

        start2 = end1
        end2 = [end1[0] + segmentsSize, end1[1] + segmentsSize]

        start3 = end2
        end3 = [end2[0] + segmentsSize, end2[1] + segmentsSize]

        start4 = end3
        end4 = [end3[0] + segmentsSize, end3[1] + segmentsSize]

        for outputType in range(len(rawDataList)):
            segmentTypesList = rawDataList[outputType][0]
            output = self.__getYData__(rawDataList[outputType][1])

            for i in range(len(self.SLOW)):
                steps0 = segmentTypesList[0][i]
                firstSegment = np.linspace(start=start1, stop=end1, num=steps0, endpoint=False)

                for j in range(len(self.SLOW)):
                    steps1 = segmentTypesList[1][j]
                    secondSegment = np.linspace(start=start2, stop=end2, num=steps1, endpoint=False)

                    for k in range(len(self.SLOW)):
                        steps2 = segmentTypesList[2][k]
                        thirdSegment = np.linspace(start=start3, stop=end3, num=steps2, endpoint=False)

                        if numberOfSegments > 3:
                            for m in range(len(self.SLOW)):
                                steps3 = segmentTypesList[3][m]
                                fourthSegment = np.linspace(start=start4, stop=end4, num=steps3, endpoint=False)

                                sequence = [firstSegment, secondSegment, thirdSegment, fourthSegment, [[MAX, MAX]]]

                                path = np.concatenate(sequence)
                                path = pd.DataFrame(path, columns=['x', 'y'])

                                activation = self.__calc_activation_matrix__(path, pcs)

                                self.dataList.append((activation, output))
                        else:
                            sequence = [firstSegment, secondSegment, thirdSegment, [[MAX, MAX]]]

                            path = np.concatenate(sequence)
                            path = pd.DataFrame(path, columns=['x', 'y'])

                            activation = self.__calc_activation_matrix__(path, pcs)

                            self.dataList.append((activation, output))

    def __len__(self):
        return len(self.dataList)

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.dataList)

    def __getitem__(self, index):
        X = []
        y = []

        X.append(np.transpose(self.dataList[index][0]))
        y.append(self.dataList[index][1])

        X = np.array(X)
        y = np.array(y)

        return X, y
