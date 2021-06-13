import datetime
import math
import multiprocessing
import time
from sklearn.metrics import accuracy_score

import numpy as np
import cv2
import pandas as pd


def generate_filters():
    # Generate Gabor features
    kernels = []  # Create empty list to hold all kernels that we will generate in a loop
    for theta in range(8):  # Define number of thetas. Here only 2 theta values 0 and 1/4 . pi
        theta = theta / 4. * np.pi
        for sigma in (1, 3, 5, 7):  # Sigma with values of 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                    for ksize in [3]:
                        for phi in (0, 0.4, 0.8, 1):
                            gabor_label = f"ksize={ksize}_sigma={sigma}_theta={theta}_" \
                                          f"lambda={lamda}_gamma={gamma}_phi={phi}".replace(".", "-")
                            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi,
                                                        ktype=cv2.CV_32F)
                            if not all(all(math.isnan(elem) for elem in row) for row in kernel):
                                kernels.append([kernel, gabor_label])
    return kernels


#
# def apply_filters(x, kernels):
#     ret_frame = pd.DataFrame()
#     for i, kernel in enumerate(kernels):
#         filtered_images = []
#         print(i)
#         for image in x:
#             filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernel[0]))
#         ret_frame[i] = (filtered_images, kernel[1])
#     return ret_frame


# def apply_filters(x, kernels):
#     for i, kernel in enumerate(kernels):
#         frame = pd.DataFrame()
#         filtered_images = []
#         print(i)
#         for image in x:
#             filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernel[0]))
#         frame[i] = (filtered_images, kernel[1])
#         frame.to_pickle(f"./temp/{i}_{kernel[1]}.pkl", compression="gzip")


def apply_filters(x, kernel):
    filtered_images = []
    for j, image in enumerate(x):
        filtered_images.append(cv2.filter2D(image, cv2.CV_8UC3, kernel))
    return filtered_images


def apply_filters_2d(x, kernel):
    filtered_images = []
    for j, image in enumerate(x):
        filtered_images.append(cv2.filter2D(np.reshape(image, (28, 28)), cv2.CV_8UC3, kernel))
    return filtered_images


def worker(queue: multiprocessing.Queue, lock, destination_folder):
    """
    A function performed by a Process. It carries out the tasks until the queue is empty.

    :param queue: multiprocessing.Queue. Holds tasks performed by workers.
    """
    while True:
        if queue.empty():
            return
        current_task = queue.get()
        data, kernel, label = current_task[1]
        filtered_images = current_task[0](data, kernel[0])
        print(kernel[1])
        np.savez_compressed(f"{destination_folder}/{label}_{kernel[1]}.npz", filtered_images)


def add_starting_objects_to_queue(queue: multiprocessing.Queue, kernels, data, label):
    """
    Adds all starting tasks to the queue.

    :param queue: multiprocessing.Queue. Holds tasks performed by workers.
    :param chess_board: Board. A chess board that the computations are being performed at.
    :param depth: a number that signifies how deep the search will be.
    """
    for kernel in kernels:
        queue.put((apply_filters_2d, (data, kernel, label)))


def multiprocess_perft_starter(number_of_processes: int, kernels, data, label, destination_folder='./temp2'):
    """
    Initializes all key components to perform a perft operation.

    :param chess_board: Board. A chess board that the computations are being performed at.
    :param depth: a number that signifies how deep the search will be.
    :param number_of_processes: a number of Processes that will perform a perft operation.
    :return: number of all possible positions positions for a given chess_board and depth.
    """
    processes = []
    queue = multiprocessing.Queue()
    add_starting_objects_to_queue(queue, kernels, data, label)
    lock = multiprocessing.Lock()
    for i in range(number_of_processes):
        process = multiprocessing.Process(target=worker, args=(queue, lock, destination_folder), name=f"Process {i}")
        processes.append(process)
        process.start()
    for p in processes:
        p.join()


def select_best_kernel(y_train, y_test, folder_path, labels, model_constructor):
    data_frame = pd.DataFrame({"Kernel Label": [], "Accuracy": []})
    for label in labels:
        x_train = np.reshape(np.load(f"{folder_path}/train_{label}.npz")["arr_0"], (len(y_train), 784))
        x_test = np.reshape(np.load(f"{folder_path}/test_{label}.npz")["arr_0"], (len(y_test), 784))
        model = model_constructor()
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        data_frame = data_frame.append({"Kernel Label": label, "Accuracy": accuracy}, ignore_index=True)
    return data_frame


def get_kernel_parameters_from_label(label):
    label = label.replace("\n", "")
    params = label.split("_")
    param_values = list(map((lambda x: float(x.split("=")[1].replace("-", "."))), params))
    param_values = tuple([(int(param_values[0]), int(param_values[0]))] + param_values[1:])
    return param_values
