from plotting import ROCPlotter
import matplotlib.pyplot as plt
import numpy as np


# create some mock data to plot
x_list = []
y_list = []

# create an instance of the ROCPlotter class
roc_plotter = ROCPlotter()

for i in range(10):
    x_list.append(i)
    y_list.append(i)

    # plot the data
    roc_plotter.plot_roc_curve(x_list, y_list, i)

