import pandas as pd
import numpy as np

from Data import ReadData
from FeatureExtraction import Gabor, FeatureExtractors
from Models import NaiveBayes, KNN


def some_statistics(path_to_data_frame, path_to_data):
    df = pd.read_pickle(path_to_data_frame)
    df = pd.DataFrame.sort_values(df, by="Accuracy", ascending=False)
    print("all")
    print(df.to_string())
    best_kernel_label = df.set_index('Kernel Label').idxmax()[0]
    print("best gabor kernel: ", best_kernel_label.replace("-", ".").replace("_", " "))
    _, y_train = ReadData.load_mnist(path_to_data)
    _, y_test = ReadData.load_mnist(path_to_data, 't10k')

    models = []
    FeatureExtractors.initialize_kernel(*Gabor.get_kernel_parameters_from_label(best_kernel_label))
    for feature_extractor in ['gabor', 'prewitt_h', 'prewitt_v', 'sobel', 'canny']:
        x_train = FeatureExtractors.feature_extraction(ReadData.load_mnist(path_to_data, 'train')[0], feature_extractor)
        x_test = FeatureExtractors.feature_extraction(ReadData.load_mnist(path_to_data, 't10k')[0], feature_extractor)
        x_train = np.reshape(x_train, (60000, 784))
        x_test =  np.reshape(x_test, (10000, 784))
        modelNB = NaiveBayes.get_classifier(x_train, y_train)
        accuracyNB = NaiveBayes.test_classifier(x_test, y_test, modelNB)
        modelKNN = KNN.get_classifier(x_train, y_train, 3)
        accuracyKNN = KNN.test_classifier(x_test, y_test, modelKNN)
        print(f"accuracy for {feature_extractor}")
        print("KNN, k=3: ", accuracyKNN, "NaiveBayes: ", accuracyNB)
        models.append([modelKNN, modelNB])
