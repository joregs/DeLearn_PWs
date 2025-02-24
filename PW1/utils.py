import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def read_data(data_type='FashionMNIST', storage_path='data'):
    """
    reads MNIST or FashionMNIST data to folder 'data' and gives back images and labels
    :param data_type: 'MNIST' or 'FashionMNIST' (<- default)
    :param storage_path: path to store data in (default 'data')
    :return:
        x: list of images (70'000) of size 28 x 28 (single channel)
        y: list of labels (0 .. 9)
        labels_map: dict of strings designating the label
    """
    # only at first execution data is downloaded, because it is saved in subfolder ./data;
    if data_type == 'MNIST':
        training_data = torchvision.datasets.MNIST(
            root=storage_path,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        test_data = torchvision.datasets.MNIST(
            root=storage_path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        # labels for MNIST (just for compatibility reasons)
        labels_map = {
            0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
            9: "Nine",
        }
    else:
        training_data = torchvision.datasets.FashionMNIST(
            root=storage_path,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        test_data = torchvision.datasets.FashionMNIST(
            root=storage_path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        # labels for FashionMNIST
        labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }

    # convert to numpy array (originally it is a torch.tensor)
    x = training_data.data.numpy()
    x = np.append(x, test_data.data.numpy(), 0)

    y = training_data.targets.numpy()
    y = np.append(y, test_data.targets.numpy())

    print(f'data is {data_type}, shape of x is {x.shape}\n , shape of y is {y.shape}')

    return x, y, labels_map


def plot_img(img, col_map=plt.cm.gray, figure_size=[3, 3]):
    """
    plot a single mnist image
    :param img: input image
    :param col_map: color map, (default plt.cm.gray)
    :param figure_size: figure size, (default [3,3])
    :return:
    """
    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()
    ax.imshow(img, cmap=col_map)
    ax.set_axis_off()
    plt.show()


def plot_tiles(x_array, rows, cols=-1, figure_size=[10, 10]):
    """
    plot list of images as single image
    :param x_array: array of images (being organised as ROWS!)
    :param rows/cols: an image of rows x cols - images is created (if x_array is smaller zeros are padded)
    :param figure_size: size of full image created (default [10,10])
    :return:
    """

    digit_size = 28  # size of digit (width = height)

    # use rows = cols as default
    if cols < 0:
        cols = rows

    if x_array.shape[0] < rows * cols:
        cols = int(x_array.shape[0] / rows)
        remain = np.mod(x_array.shape[0], rows)
        if 0 < remain:
            cols += 1
            x_array = np.append(x_array, np.zeros((rows - remain, x_array.shape[1])), 0)

    img = x_array[0:rows, :].reshape(rows * digit_size, digit_size)
    for i0 in range(1, cols):
        # the reshape operator in the append call takes num of digit_size x digit_size images and
        # puts them in a single column; append then does the rest
        img = np.append(img, x_array[i0 * rows:(i0 + 1) * rows, :].reshape(rows * digit_size, digit_size), 1)

    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()


def plot_error(nn_instance, figure_size=[6, 6]):
    """
    analyse error as function of epochs
    :param nn_instance: NeuralNetwork class to plot
    :param figure_size: size of full image created (default [6,6])
    :return:
    """
    epochs = np.arange(nn_instance.epoch_counter)
    train_error = nn_instance.result_data[:, 1]
    val_error = nn_instance.result_data[:, 3]

    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()
    ax.semilogy(epochs, train_error, label="train")
    ax.semilogy(epochs, val_error, label="test")
    ax.set_ylabel('Error')
    ax.set_xlabel('Epochs')
    xmax = epochs[-1]
    ymin = 5e-4
    ymax = 5e-1
    ax.axis([0, xmax, ymin, ymax])
    ax.legend()
    plt.show()


def plot_cost(nn_instance, figure_size=[6, 6]):
    """
    analyse cost as function of epochs
    :param nn_instance: NeuralNetwork class to plot
    :param figure_size: size of full image created (default [6,6])
    :return:
    """
    epochs = np.arange(nn_instance.epoch_counter)
    train_costs = nn_instance.result_data[:, 0]
    val_costs = nn_instance.result_data[:, 2]

    fig = plt.figure(figsize=figure_size)
    ax = fig.subplots()
    ax.semilogy(epochs, train_costs, label="train")
    ax.semilogy(epochs, val_costs, label="test")
    ax.set_ylabel('Cost')
    ax.set_xlabel('Epochs')
    xmax = epochs[-1]
    ymin = 5e-3
    ymax = 2
    ax.axis([0, xmax, ymin, ymax])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    x, y, labels_map = read_data()
    print(x.shape)
