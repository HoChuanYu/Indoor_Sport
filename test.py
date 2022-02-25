import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# max_height, min_height 比 range 的 1/10 大 or 小 就要求調整

max_height = []
min_height = []
max_period = 0

data = np.asanyarray(pd.read_csv('sequence.txt',header=None))
for arr in data:
    h_temp=[]
    l_temp=[]
    # print(max_height(arr)-min_height(arr))
    h_peaks, _ = find_peaks(arr, height=np.average(arr))
    l_peaks, _ = find_peaks(arr*(-1), height=-np.average(arr))
    print('high:',h_peaks)
    print('low:',l_peaks)
    max_height.append(np.average(np.asanyarray(arr)[h_peaks]))
    min_height.append(np.average(np.asanyarray(arr)[l_peaks]))
    for i in range(len(h_peaks)-1):
        h_temp.append(h_peaks[i+1]-h_peaks[i])
    for i in range(len(l_peaks)-1):
        l_temp.append(l_peaks[i+1]-l_peaks[i])
    
    h_period = np.average(np.asanyarray(h_temp)) if len(h_temp)>3 else 0
    l_period = np.average(np.asanyarray(l_temp)) if len(l_temp)>3 else 0
    
    max_period = max(max_period,h_period,l_period)

# pick the longest period as the total period(提醒要做快點or慢點的指標) 
    # print('h_period:',h_temp)
    # print('l_period:',l_temp,'\n')

print('period',max_period)
print(max_height)
print(min_height)

file = open('Jump.txt','w')
file.write(str(max_period)+'\n')
file.write((','.join(str(a) for a in max_height)) +'\n')
file.write((','.join(str(a) for a in min_height)) +'\n')

file.close()