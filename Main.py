import os
import pickle

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import np_utils

from Data import ReadData
from FeatureExtraction import FeatureExtractors, Gabor
from Models import NaiveBayes, KNN, NeuralNetwork
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from Statistics import Statistics

# if __name__ == '__main__':
#     Statistics.some_statistics('./kernels_tested2', './Data')
# print(data_frame.set_index('Kernel Label').idxmax()[0])
# x_train, y_train = ReadData.load_mnist(path="Data")
# x_test, y_test = ReadData.load_mnist(path="Data", kind="t10k")
# df = pd.read_pickle("kernels_tested2")
# print(df.set_index('Kernel Label').idxmax()[0])
# df = pd.DataFrame.sort_values(df, by="Accuracy", ascending=False)
# print(df.to_string())
# params = Gabor.get_kernel_parameters_from_label("ksize=3_sigma=3_theta=3-9269908169872414_"
#                                                 "lambda=0-7853981633974483_gamma=0-5_phi=0-4")
# kernel = cv2.getGaborKernel(*params, ktype=cv2.CV_32F)
# Gabor.select_best_kernel(y_train, y_test, "./temp", np.array(Gabor.generate_filters())[:,1], GaussianNB)
# x_train1 = Gabor.apply_filters(x_train, kernel)
# x_test1 = Gabor.apply_filters(x_test, kernel)
# x_train2 = Gabor.apply_filters_2d(x_train, kernel)
# x_test2 = Gabor.apply_filters_2d(x_test, kernel)
# model = KNN.get_classifier(x_train, y_train, 5)
# print(KNN.test_classifier(x_test, y_test, model))
# model1 = NaiveBayes.get_classifier(np.reshape(x_train1, (60000, 784)), y_train)
# print(NaiveBayes.test_classifier(np.reshape(x_test1, (10000, 784)), y_test, model1))
# model2 = NaiveBayes.get_classifier(np.reshape(x_train2, (60000, 784)), y_train)
# print(NaiveBayes.test_classifier(np.reshape(x_test2, (10000, 784)), y_test, model2))
# img = cv2.imread("Untitleda.png", cv2.IMREAD_GRAYSCALE)
# df = Gabor.select_best_kernel(y_train, y_test, "./temp2", np.array(Gabor.generate_filters())[:,1], GaussianNB)
# df.to_pickle("kernels_tested2")
# model = NaiveBayes.get_classifier(x_train, y_train)
# acc = NaiveBayes.test_classifier(x_test, y_test, model)
# x = FeatureExtractors.feature_extraction(x_train, "sobel")
# x2 = FeatureExtractors.feature_extraction(x_test, "sobel")
# model2 = NaiveBayes.get_classifier(x, y_train)
# acc2 = NaiveBayes.test_classifier(x2, y_test, model2)
#
# print(acc, acc2)
# filters = Gabor.generate_filters()
# print(len(filters))
# print("alleluyah")
# Gabor.multiprocess_perft_starter(4, filters, x_train, "train")
# Gabor.multiprocess_perft_starter(4, filters, x_test, "test")
# Gabor.apply_filters(x_train, filters, "train")
# Gabor.apply_filters(x_test, filters, "test")
# for i in range(4096):
#     xd = filtered_images[i]
#     path = os.path.join(os.curdir, "temp", f"{xd[1]}.png")
#     print(path)
#     print(cv2.imwrite(path, np.reshape(xd[0], (28,28))))
#
# filtered_images.to_pickle("filtered_x_train2.pkl")
# filtered_images2 = Gabor.apply_filters(x_test, filters[0::50])
# filtered_images2.to_pickle("filtered_x_test.pkl")

if __name__ == '__main__':
    for fe in ["prewitt_h", "prewitt_v", "sobel", "canny", "gabor"]:
        x_train, y_train = ReadData.load_mnist(path="Data")
        x_test, y_test = ReadData.load_mnist(path="Data", kind="t10k")
        x_train = FeatureExtractors.feature_extraction(x_train, fe)
        x_test = FeatureExtractors.feature_extraction(x_test, fe)
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
        # y_train = np_utils.to_categorical(y_train, 10)
        # y_test = np_utils.to_categorical(y_test, 10)
        # x_train = np.array(x_train).astype('float32')
        # x_test = np.array(x_test).astype('float32')
        # x_train /= 255
        # x_test /= 255
        # x_train += 1
        # x_train /= 2
        # x_test += 1
        # x_test /= 2
        # model = NeuralNetwork.create_model2(225, 4)
        # NeuralNetwork.train_model(model, x_train, y_train, x_test, y_test, 10, 'ModelNN5')
        model = NaiveBayes.get_classifier(x_train, y_train)
        # Open a file and use dump()
        with open(f'Models/TrainedModels/Bayes/Bayes_{fe}.pkl', 'wb') as file:
            print(NaiveBayes.test_classifier(x_test, y_test, model))
            pickle.dump(model, file)
