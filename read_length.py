from os import read
import numpy as np
from pandas import read_csv

data = read_csv('data_csv/vbf_data_2018.csv', header=None)
print('number of lines:', len(data))