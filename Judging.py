import threading
import vlc
from matplotlib.pyplot import pause
import PoseModule as pm
import pandas as pd
import numpy as np
import math
import time
from scipy.signal import find_peaks
import cv2
import matplotlib.pyplot as plt

Landmarkers = {'right_shoulder':[], 'left_shoulder':[],
                'right_elbow':[], 'left_elbow':[],
                'right_thigh':[], 'left_thigh':[],
                'right_knee':[], 'left_knee':[]}

Voice = ['faster','slower','lower','higher','wider']

def thread_func(*name):
    # print(name)
    for i in range(len(name)):
        s = 'Voices/'+str(Voice[name[i]])+'.m4a'
        # print(s)
        p = vlc.MediaPlayer(s)
        p.play()
        time.sleep(2.5)           # play .mp3
    # print('')

if __name__ == '__main__':
    data = np.asanyarray(pd.read_csv('Jump.txt',header=None,sep=' '))

    period = float(data[0][0])
    max_height = list(np.float_(str(data[0][1]).split(',')))
    min_height = list(np.float_(str(data[0][2]).split(',')))
    joint_range = [max_height[i]-min_height[i] for i in range(8)]
    # print(data[0][2])
    

    cap = cv2.VideoCapture(0)#'Videos/video3.mov')
    detector = pm.PoseDetector()
    thr = threading.Thread()

    pTime = 0
    cutoff = 2.667
    max_period=0
    pre_len = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        try:
            detector.findPose(img,draw=False)
            lmList = detector.findPosition(img,draw=False)
        except:
            break
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime


        JointAngleCos = detector.findJointCosAngle(lmList=lmList)
        if len(Landmarkers['right_shoulder']) > 100 and len(JointAngleCos) != 8:
            break
        Landmarkers['right_shoulder'].append(JointAngleCos[0])
        Landmarkers['left_shoulder'].append(JointAngleCos[1])
        Landmarkers['right_elbow'].append(JointAngleCos[2])
        Landmarkers['left_elbow'].append(JointAngleCos[3])
        Landmarkers['right_thigh'].append(JointAngleCos[4])
        Landmarkers['left_thigh'].append(JointAngleCos[5])
        Landmarkers['right_knee'].append(JointAngleCos[6])
        Landmarkers['left_knee'].append(JointAngleCos[7])

        all_h_peaks=[]
        all_l_peaks=[]
        for key in Landmarkers.keys():
            x=np.arange(len(Landmarkers[key]))+1
            y=Landmarkers[key][:]
            minus_y = [i*(-1) for i in y]
            h_peaks, _ = find_peaks(y, height=np.average(y))
            l_peaks, _ = find_peaks(minus_y, height=-np.average(y))
            all_h_peaks.append(h_peaks)
            all_l_peaks.append(l_peaks)

            if key == 'right_shoulder':
                judge_arr=h_peaks.copy()
            
                h_period = h_peaks[len(h_peaks)-1]-h_peaks[len(h_peaks)-2] if len(judge_arr)>1 else 0
                l_period = l_peaks[len(l_peaks)-1]-l_peaks[len(l_peaks)-2] if len(judge_arr)>1 else 0
                
            max_period = max(h_period,l_period)

        video_len = len(Landmarkers['right_shoulder'])
        cur_len = len(judge_arr)

        if cur_len>1 and pre_len!=cur_len and cur_len%4==0:
            voice_arr = []
            # print(period, max_period)
            if max_period>(period*1.2):
                voice_arr.append(0)                 # print('做快點!')
            elif max_period<(period*0.8):
                voice_arr.append(1)                 # print('做慢點!')

            for i in range(8):
                if i==0:
                    # print(Landmarkers['right_shoulder'])
                    # print('shoulder:\n',np.asanyarray(Landmarkers['right_shoulder'])[all_h_peaks[0]])
                    # print(np.asanyarray(Landmarkers['right_shoulder'])[all_l_peaks[0]])
                    # print(max_height[0],min_height[0])
                    if len(all_h_peaks[0])>0:
                        if Landmarkers['right_shoulder'][all_h_peaks[0][len(all_h_peaks[0])-1]] < (max_height[0]-joint_range[0]*0.1):
                            voice_arr.append(2)                 # print('手放低')
                    if len(all_l_peaks[0])>0:
                        if Landmarkers['right_shoulder'][all_l_peaks[0][len(all_l_peaks[0])-1]] > (min_height[0]+joint_range[0]*0.1):
                            voice_arr.append(3)                 # print('手舉高')
                        
                elif i==4:
                    # print(Landmarkers['right_thigh'])
                    # print('thigh:',np.asanyarray(Landmarkers['right_thigh'])[video_len-1-math.ceil(max_period):video_len-1])
                    # print(min_height[4])
                    
                    # Landmarkers['right_thigh'][all_l_peaks[4][len(all_l_peaks[4])-1]]
                    if len(all_l_peaks[4])>0:
                        # print(min(Landmarkers['right_thigh'][video_len-1-math.ceil(max_period):video_len-1]),min_height[4])
                        if min(Landmarkers['right_thigh'][video_len-1-math.ceil(max_period):video_len-1]) > (min_height[4]+joint_range[4]*0.2): 
                            voice_arr.append(4)                 # print('腳打開')
        
            thr = threading.Thread(target=thread_func, args=(voice_arr))
            thr.start()

        pre_len=cur_len
        img_r = cv2.resize(img,(300,640))
        cv2.imshow("Image",img_r)
        cv2.waitKey(1)

    thr.join()

    count=0
    fig, ax = plt.subplots(2,4,sharex=True,sharey=True)
    fig.suptitle("LandMark(Cos Angle)",fontsize=16)
    for key in Landmarkers.keys():
       x=np.arange(len(Landmarkers[key]))+1
       y=Landmarkers[key]
       ax[int(count/4)][int(count%4)].plot(x,y)
       ax[int(count/4)][int(count%4)].set_title(key)
       count+=1
    fig.savefig('Judge.png')