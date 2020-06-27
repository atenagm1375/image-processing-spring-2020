import os

import numpy as np


DIR = '../../ISICArchive/'

downloaded = []
for file in os.listdir(DIR):
    if file[-3:] == 'jpg':
        number = int(file.split('_')[1][:-4])
        downloaded.append(number)

print("SKIPPED NUMBERS:")
for i in range(np.min(downloaded), np.max(downloaded)):
    if i not in downloaded:
        print(i)
