# 需要解決若landmark被遮住的情況，現在的code只要landmark被遮住就會有error
# scipy.signal.find_peaks
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import time
from scipy.signal import butter, lfilter, find_peaks
import PoseModule as pm
import matplotlib.pyplot as plt

Landmarkers = {'right_shoulder':[], 'left_shoulder':[],
                'right_elbow':[], 'left_elbow':[],
                'right_thigh':[], 'left_thigh':[],
                'right_knee':[], 'left_knee':[]}

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    


if __name__ == '__main__':
    cap = cv2.VideoCapture("Videos/video2.mp4")
    detector = pm.PoseDetector()

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        JointAngleCos = detector.findJointCosAngle(lmList=lmList)
        Landmarkers['right_shoulder'].append(JointAngleCos[0])
        Landmarkers['left_shoulder'].append(JointAngleCos[1])
        Landmarkers['right_elbow'].append(JointAngleCos[2])
        Landmarkers['left_elbow'].append(JointAngleCos[3])
        Landmarkers['right_thigh'].append(JointAngleCos[4])
        Landmarkers['left_thigh'].append(JointAngleCos[5])
        Landmarkers['right_knee'].append(JointAngleCos[6])
        Landmarkers['left_knee'].append(JointAngleCos[7])

        cv2.imshow("Image",img)
        cv2.waitKey(1)
    
file = open(r"LandMarks.txt","w+")
file.write(str(Landmarkers))
file.close()


fig, ax = plt.subplots(2,4,sharex=True,sharey=True)
fig.suptitle("LandMark(Cos Angle)",fontsize=16)
fig_1, ax_1 = plt.subplots(2,4,sharex=True,sharey=True)
fig_1.suptitle("LandMark(Cos Angle)_Smoothing",fontsize=16)
count = 0
cutoff = 2.667

data=[]
p_array=[]
h_array=[]
for key in Landmarkers.keys():
    x=np.arange(len(Landmarkers[key]))+1
    y=Landmarkers[key]
    ax[int(count/4)][int(count%4)].plot(x,y)
    ax[int(count/4)][int(count%4)].set_title(key)

    y_filt = butter_lowpass_filter(y[10:len(y)-10],cutoff,fps)              # 將開始尚未動作以及結束時的動作去掉
    ax_1[int(count/4)][int(count%4)].plot(x[:len(y)-30],y_filt[10:])        # 把一開始的突入波去掉
    ax_1[int(count/4)][int(count%4)].set_title(key)
    data.append(y_filt)
    count+=1
fig.savefig('Landmark.png')
fig_1.savefig('Smoothing.png')



max_height = []
min_height = []
max_period = 0

for arr in data:
    h_temp=[]
    l_temp=[]
    # print(max_height(arr)-min_height(arr))
    h_peaks, _ = find_peaks(arr, height=np.average(arr))
    l_peaks, _ = find_peaks(arr*(-1), height=-np.average(arr))
    # print('high:',h_peaks)
    # print('low:',l_peaks)
    max_height.append(np.average(np.asanyarray(arr)[h_peaks]))
    min_height.append(np.average(np.asanyarray(arr)[l_peaks]))
    for i in range(len(h_peaks)-1):
        h_temp.append(h_peaks[i+1]-h_peaks[i])
    for i in range(len(l_peaks)-1):
        l_temp.append(l_peaks[i+1]-l_peaks[i])
    
    h_period = np.average(np.asanyarray(h_temp)) if len(h_temp)>3 else 0
    l_period = np.average(np.asanyarray(l_temp)) if len(l_temp)>3 else 0
    
    max_period = max(max_period,h_period,l_period)


# print('period',max_period)
# print(max_height)
# print(min_height)

file = open('Jump.txt','w')
file.write(str(max_period)+'\n')
file.write((','.join(str(a) for a in max_height)) +'\n')
file.write((','.join(str(a) for a in min_height)) +'\n')

file.close()