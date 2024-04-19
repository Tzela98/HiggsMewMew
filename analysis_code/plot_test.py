import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
import mplhep as hep
from torch import rand


def generate_random_list(length):
    random_list = [random() for _ in range(length)]
    return random_list


hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)

random_list_1 = np.random.normal(10, 3, 400)
random_list_2 = np.random.normal(12, 3, 400)

h1, bins1 = np.histogram(random_list_1, bins=10, range=(0, 25))
h2, bins2 = np.histogram(random_list_2, bins=10, range=(0, 25))

print(h1, h2)

hep.histplot([h1, h2], bins1, stack=True, label="Data")

plt.show()