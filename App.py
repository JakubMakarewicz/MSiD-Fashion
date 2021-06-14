import os
import pickle
import sys
from functools import partial

# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow
from keras.utils import np_utils

from Data import ReadData
from FeatureExtraction import Gabor, FeatureExtractors
from Data.ReadData import load_mnist
from Models import KNN, NaiveBayes, NeuralNetwork


def save_gabor_kernels(file_name):
    np.savez_compressed(file_name, np.array(Gabor.generate_filters()))


def apply_gabor_kernels(gabor_file, images_file, label, destination_folder, threads, fashion_path=None):
    kernels = np.load(gabor_file)
    if fashion_path is not None:
        data = load_mnist(fashion_path, label)
    else:
        data = np.load(images_file)
    data = np.reshape(data, (data.shape[0], -1))
    Gabor.multiprocess_perft_starter(threads, kernels, data, label, destination_folder)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

model_constructors = {'KNN': KNeighborsClassifier,
                      'NB':  GaussianNB}


def find_best_gabor_kernel(x_path, y_train_path, y_test_path, gabor_file, model):
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    kernels = np.load(gabor_file)[:, 1]
    df = Gabor.select_best_kernel(y_train, y_test, x_path, kernels, model_constructors[model])
    print(df.set_index('Kernel Label').idxmax()[0])


feature_extractors = {2: 'prewitt_h',
                      3: 'prewitt_v',
                      4: 'sobel',
                      5: 'canny',
                      6: 'gabor'}

labels = {0: " 	T-shirt/top",
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"}

if __name__ == '__main__':
    end = False
    while not end:
        feature_extractor = int(input("please select a feature extractor:\n"
                                      "1. None\n"
                                      "2. Prewitt_H\n"
                                      "3. Prewitt_V\n"
                                      "4. Sobel\n"
                                      "5. Canny\n"
                                      "6. Gabor\n"))
        option = int(input("please select an option:\n"
                           "1. Train a model\n"
                           "2. Test the accuracy of a model\n"
                           "3. Make a prediction\n"))
        if option in [1, 2]:
            data_path = input("please input the path to fashion-mnist data: ")
            x_train, y_train = ReadData.load_mnist(data_path)
            x_test, y_test = ReadData.load_mnist(data_path, "t10k")

            if feature_extractor in [2, 3, 4, 5]:
                x_train = FeatureExtractors.feature_extraction(x_train, feature_extractors[feature_extractor])
                x_test = FeatureExtractors.feature_extraction(x_test, feature_extractors[feature_extractor])
            elif feature_extractor == 6:
                choice = int(input("Do you want to use the default kernel "
                                   "(ksize=3 sigma=5 theta=3.9269908169872414 lambda=0.7853981633974483 gamma=0.5 phi=0.4)?:\n"
                                   "1: Yes\n"
                                   "2: No\n"))
                if choice == 2:
                    ksize = int(input("Ksize: "))
                    sigma = int(input("sigma: "))
                    theta = int(input("theta: "))
                    lambd = int(input("lambda: "))
                    gamma = int(input("gamma: "))
                    phi = int(input("phi: "))
                    FeatureExtractors.initialize_kernel(False, ((ksize, ksize), sigma, theta, lambd, gamma, phi))
                else:
                    FeatureExtractors.initialize_kernel(True)
                x_train = FeatureExtractors.feature_extraction(x_train, feature_extractors[feature_extractor])
                x_test = FeatureExtractors.feature_extraction(x_test, feature_extractors[feature_extractor])

            model_type = int(input("please select a model type:\n"
                                   "1. KNN\n"
                                   "2. NaiveBayes\n"
                                   "3. NeuralNetwork\n"))
            if model_type in [1, 2]:
                x_train = np.reshape(x_train, (-1, 784))
                x_test = np.reshape(x_test, (-1, 784))
            elif model_type == 3:
                if feature_extractor == 1:
                    y_train = np_utils.to_categorical(y_train, 10)
                    y_test = np_utils.to_categorical(y_test, 10)
                    x_train = np.array(x_train).astype('float32')
                    x_test = np.array(x_test).astype('float32')
                    x_train /= 255
                    x_test /= 255
                else:
                    y_train = np_utils.to_categorical(y_train, 10)
                    y_test = np_utils.to_categorical(y_test, 10)
                    x_train += 1
                    x_train /= 2
                    x_test += 1
                    x_test /= 2
                x_train = np.reshape(x_train, (-1, 28, 28, 1))
                x_test = np.reshape(x_test, (-1, 28, 28, 1))

            if option == 1:
                if model_type == 1:
                    model = KNN.get_classifier(x_train, y_train, int(input("how many neighbors?: ")))
                    with open(f'{input("destination path: ")}.pkl', 'wb') as file:
                        pickle.dump(model, file)
                elif model_type == 2:
                    model = NaiveBayes.get_classifier(x_train, y_train)
                    with open(f'{input("destination path: ")}.pkl', 'wb') as file:
                        pickle.dump(model, file)
                elif model_type == 3:
                    which = int(input("Please select the NN type:\n"
                                      "1. basic \n"
                                      "2. custom convolutional\n"))
                    if which == 1:
                        model = NeuralNetwork.create_model1()
                    else:
                        model = NeuralNetwork.create_model2(int(input("convolutional layer size: ")),
                                                            int(input("how many convolutional layers?: ")))
                    NeuralNetwork.train_model(model, x_train, y_train, x_test, y_test,
                                              int(input("epochs: ")), input("destination path: "))
            elif option == 2:
                model_path = input("Please input the path to the model: ")
                if model_type in [1, 2]:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    if model_type == 1:
                        accuracy = KNN.test_classifier(x_test, y_test, model)
                    else:
                        accuracy = NaiveBayes.test_classifier(x_test, y_test, model)
                else:
                    model = tensorflow.keras.models.load_model(f'{model_path}')
                    accuracy = NeuralNetwork.test_model(model, x_test, y_test)
                print(accuracy)
        if option == 3:
            model_type = int(input("please select a model type:\n"
                                   "1. KNN\n"
                                   "2. NaiveBayes\n"
                                   "3. NeuralNetwork\n"))
            model_path = input("Please input the path to the model: ")
            if model_type in [1, 2]:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = tensorflow.keras.models.load_model(f'{model_path}')

            img = cv2.imread(input("path to image: "), cv2.IMREAD_GRAYSCALE)

            if feature_extractor in [2, 3, 4, 5]:
                img = FeatureExtractors.feature_extraction([img], feature_extractors[feature_extractor])
            elif feature_extractor == 6:
                choice = int(input("Do you want to use the default kernel "
                                   "(ksize=3 sigma=5 theta=3.9269908169872414 lambda=0.7853981633974483 gamma=0.5 phi=0.4)?:\n"
                                   "1: Yes\n"
                                   "2: No\n"))
                if choice == 2:
                    ksize = int(input("Ksize: "))
                    sigma = int(input("sigma: "))
                    theta = int(input("theta: "))
                    lambd = int(input("lambda: "))
                    gamma = int(input("gamma: "))
                    phi = int(input("phi: "))
                    FeatureExtractors.initialize_kernel(False, ((ksize, ksize), sigma, theta, lambd, gamma, phi))
                else:
                    FeatureExtractors.initialize_kernel(True)
                img = FeatureExtractors.feature_extraction([img], feature_extractors[feature_extractor])

            if model_type in [1, 2]:
                img = np.reshape(img, (-1, 784))
                print(labels[model.predict(img)[0]])
            elif model_type == 3:
                if feature_extractor == 1:
                    img = np.array(img).astype('float32')
                    img /= 255
                else:
                    img += 1
                    img /= 2
                img = np.reshape(img, (-1, 28, 28, 1))
                print(labels[int(np.argmax(model.predict(img)))])
