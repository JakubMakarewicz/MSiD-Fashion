import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

# if __name__ == '__main__':
# images = load_mnist("../Dataa")
# ksize = 3  # Use size that makes sense to the image and fetaure size. Large may not be good.
# # On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
# sigma = 5  # Large sigma on small features will fully miss the features.
# theta = 1 * np.pi / 2  # /4 shows horizontal 3/4 shows other horizontal. Try other contributions
# lamda = 1 * np.pi / 4  # 1/4 works best for angled.
# gamma = 0.9  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
# # Value of 1, spherical may not be ideal as it picks up features from other regions.
# phi = 0.8  # Phase offset. I leave it to 0. (For hidden pic use 0.8)
#
# kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
#
# plt.imshow(kernel)
#
# # img = cv2.imread('images/synthetic.jpg')
# img = cv2.imread('../Untitled.png')  #Image source wikipedia: https://en.wikipedia.org/wiki/Plains_zebra
# img = np.reshape(images[0][0], (28, 28)) # USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
# # plt.imshow(img)
# # plt.show()
#
# fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
#
# kernel_resized = cv2.resize(kernel, (400, 400))  # Resize image
#
# # plt.imshow(kernel_resized)
# # plt.show()
# plt.imshow(fimg, cmap='gray')
# plt.show()
# pass
from Data import ReadData
from FeatureExtraction import FeatureExtractors, Gabor
from Models import NaiveBayes
import pandas as pd
from sklearn.naive_bayes import GaussianNB
if __name__ == '__main__':

    # print(data_frame.set_index('Kernel Label').idxmax()[0])
    df = pd.read_pickle("kernels tested")
    print(df.set_index('Kernel Label').idxmax()[0])
    x_train, y_train = ReadData.load_mnist(path="Data")
    x_test, y_test = ReadData.load_mnist(path="Data", kind="t10k")
    df = Gabor.select_best_kernel(y_train, y_test, "./temp", np.array(Gabor.generate_filters())[:,1], GaussianNB)
    # df.to_pickle("kernels tested")
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
    # Gabor.multiprocess_perft_starter(6, filters, x_train, "train")
    # Gabor.multiprocess_perft_starter(6, filters, x_test, "test")
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
