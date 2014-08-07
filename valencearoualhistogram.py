from utils.read import csv_2_dict_va
import matplotlib.pyplot as plt
import numpy as np

'''
algorthm to check distribution of 
valence and arousal
'''

valence, arousal = csv_2_dict_va('csv/survery2dataMin1.csv')

valence_hist = [0 for i in range(20)]
arousal_hist = [0 for i in range(20)]

for (key, value) in valence.items():
    for val in value:
        valence_hist[int((val+1)*10)] += 1

for (key, value) in arousal.items():
    for val in value:
        arousal_hist[int((val+1)*10)] += 1

print valence_hist
print arousal_hist

plt.subplot(2, 1, 1)
plt.bar(range(20), np.array(valence_hist))

plt.subplot(2, 1, 2)
plt.bar(range(20), np.array(arousal_hist))
plt.show()