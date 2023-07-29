import os
from math import sqrt

import numpy as np
import pandas as pd
import tqdm as tqdm
import cv2
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

PC_VALUE_AT_RADIUS = 0.6  # 0.2 activation value (constant K in pc activation formula)

TRAIN = 0
VALIDATION = 1
ALL = 2

START = -1.0
END = 1.0
STEP = 0.5

SEGMENT_SPACE = np.linspace(start=-1.0, stop=1.0, num=int((END - START) / STEP) + 1, endpoint=True)

CREATE_VIDEOS = False


class DataGenerator(keras.utils.Sequence):
    def __init__(self, freedom=2, mode=TRAIN):
        self.MAX_X = SEGMENT_SPACE
        self.MAX_Y = SEGMENT_SPACE
        self.MIN_X = SEGMENT_SPACE
        self.MIN_Y = SEGMENT_SPACE
        self.video = 0

        if mode == TRAIN:
            self.FAST = [4, 6]
            self.MEDIUM = [8, 10]
            self.SLOW = [12, 14]
            self.VERY_SLOW = [16, 18]
        elif mode == VALIDATION:
            self.FAST = [5]
            self.MEDIUM = [9]
            self.SLOW = [13]
            self.VERY_SLOW = [17]
        elif mode == ALL:
            self.FAST = [4, 5, 6]
            self.MEDIUM = [8, 9, 10]
            self.SLOW = [12, 13, 14]
            self.VERY_SLOW = [16, 17, 18]
        else:
            self.FAST = []
            self.MEDIUM = []
            self.SLOW = []
            self.VERY_SLOW = []
            self.MAX_X = []
            self.MAX_Y = []
            self.MIN_X = []
            self.MIN_Y = []

        self.train = (mode == TRAIN)

        self.freedom = freedom

        self.paths = []

        self.__create_paths__()

        if mode == TRAIN:
            self.on_epoch_end()

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

        if not self.train:
            reshaped_arr = activations.reshape(16, 16, activations.shape[1])

            reshaped_arr = np.transpose(reshaped_arr, (1, 0, 2))

            # Define video properties
            fps = 10  # Frames per second
            frame_width = reshaped_arr.shape[1]
            frame_height = reshaped_arr.shape[0]

            if CREATE_VIDEOS:
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(f"./videos/{self.freedom}/{self.video}.mp4", fourcc, fps, (frame_width, frame_height))

                for i in range(reshaped_arr.shape[2]):
                    subarray = reshaped_arr[:, :, i]
                    subarray_scaled = (subarray * 255).astype(np.uint8)

                    # Create a grayscale image from the subarray
                    frame = cv2.cvtColor(subarray_scaled, cv2.COLOR_GRAY2BGR)

                    # Write the frame to the video
                    video_writer.write(frame)

                # Release the VideoWriter
                video_writer.release()

                self.video += 1

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
        elif self.freedom == 6:
            compareData = [0, 45, 90, 135, 180, 270]

        y = [0] * len(compareData)

        for i in range(len(compareData)):
            if dataValue == compareData[i]:
                y[i] = 1
                break

        return y

    def __create_paths__(self):
        rawDataList = []
        if self.freedom == 2:
            DEGREE_90 = [self.SLOW, self.FAST, self.SLOW]
            DEGREE_270 = [self.FAST, self.SLOW, self.SLOW]
        elif self.freedom == 6:
            DEGREE_0 = [self.SLOW, self.FAST, self.MEDIUM, self.FAST]
            DEGREE_45 = [self.FAST, self.SLOW, self.MEDIUM, self.FAST]
            DEGREE_90 = [self.MEDIUM, self.FAST, self.SLOW, self.FAST]
            DEGREE_135 = [self.MEDIUM, self.SLOW, self.FAST, self.FAST]
            DEGREE_180 = [self.SLOW, self.MEDIUM, self.FAST, self.FAST]
            # DEGREE_225 = [self.FAST, self.MEDIUM, self.SLOW, self.FAST]
            DEGREE_270 = [self.FAST, self.MEDIUM, self.SLOW, self.FAST]
            # DEGREE_315 = [self.FAST, self.SLOW, self.FAST, self.FAST]
            rawDataList.append((DEGREE_0, 0))
            rawDataList.append((DEGREE_45, 45))
            rawDataList.append((DEGREE_135, 135))
            rawDataList.append((DEGREE_180, 180))
            # rawDataList.append((DEGREE_225, 225))
            # rawDataList.append((DEGREE_315, 315))

        rawDataList.append((DEGREE_90, 90))
        rawDataList.append((DEGREE_270, 270))

        self.dataList = []

        differentPaths = 0

        for max_x in tqdm.tqdm(self.MAX_X):
            for max_y in self.MAX_Y:
                for min_x in self.MIN_X:
                    for min_y in self.MIN_Y:

                        differentPaths += 1

                        numberOfSegments = len(rawDataList[0][0])
                        pathSize = sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

                        if pathSize >= 0.5:

                            pcs = pd.read_csv('input/placecells.csv', delim_whitespace=True)

                            segmentIncrement_x = (max_x - min_x) / numberOfSegments
                            segmentIncrement_y = (max_y - min_y) / numberOfSegments

                            start1 = [min_x, min_y]
                            end1 = [min_x + segmentIncrement_x, min_y + segmentIncrement_y]

                            start2 = end1
                            end2 = [end1[0] + segmentIncrement_x, end1[1] + segmentIncrement_y]

                            start3 = end2
                            end3 = [end2[0] + segmentIncrement_x, end2[1] + segmentIncrement_y]

                            start4 = end3
                            end4 = [end3[0] + segmentIncrement_x, end3[1] + segmentIncrement_y]

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
                                            thirdSegment = np.linspace(start=start3, stop=end3, num=steps2,
                                                                       endpoint=False)

                                            if numberOfSegments > 3:
                                                for m in range(len(self.SLOW)):
                                                    steps3 = segmentTypesList[3][m]
                                                    fourthSegment = np.linspace(start=start4, stop=end4, num=steps3,
                                                                                endpoint=False)

                                                    sequence = [firstSegment, secondSegment, thirdSegment,
                                                                fourthSegment,
                                                                [[max_x, max_y]]]

                                                    path = np.concatenate(sequence)
                                                    path = pd.DataFrame(path, columns=['x', 'y'])

                                                    activation = self.__calc_activation_matrix__(path, pcs)

                                                    self.dataList.append((activation, output))
                                            else:
                                                sequence = [firstSegment, secondSegment, thirdSegment,
                                                            [[max_x, max_y]]]

                                                path = np.concatenate(sequence)
                                                path = pd.DataFrame(path, columns=['x', 'y'])

                                                activation = self.__calc_activation_matrix__(path, pcs)

                                                self.dataList.append((activation, output))

        print(f'\n{differentPaths=}\n')

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
