import mediapipe as mp
import numpy as np
import math as math
import cv2

# LandMarks = ['nose','left_eye_inner','left_eye','left_eye_outer','right_eye_inner', 
#             'right_eye','right_eye_outer','','','','','','','','','','','','','','','','',
#             '','','','','','','','','','']

class PoseDetector():
    def __init__(self, mode=False, upper=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upper = upper
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()#self.mode,self.upper,self.smooth, self.detectionCon,self.trackingCon)
        
    def findPose(self, img, draw=True):         #draw body tracking
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS)

    def findPosition(self, img, draw=True):     #return landmark position
        lmList=[]
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList

    def CosAngle(self, point_1, mid_point, point_2):
        Numerator = (point_1[1]-mid_point[1])*(point_2[1]-mid_point[1])+(point_1[2]-mid_point[2])*(point_2[2]-mid_point[2])
        Denominator = math.sqrt(((point_1[1]-mid_point[1])**2 + (point_1[2]-mid_point[2])**2)*
                                ((point_2[1]-mid_point[1])**2 + (point_2[2]-mid_point[2])**2))
        CosAngle = Numerator/Denominator
        return round(CosAngle,2)

    def findJointCosAngle(self, lmList):
        JointAngleCos=[]
        JointAngleCos.append(self.CosAngle(lmList[14],lmList[12],lmList[24]))
        JointAngleCos.append(self.CosAngle(lmList[13],lmList[11],lmList[23]))
        JointAngleCos.append(self.CosAngle(lmList[12],lmList[14],lmList[16]))
        JointAngleCos.append(self.CosAngle(lmList[11],lmList[13],lmList[15]))
        JointAngleCos.append(self.CosAngle(lmList[23],lmList[24],lmList[26]))
        JointAngleCos.append(self.CosAngle(lmList[24],lmList[23],lmList[25]))
        JointAngleCos.append(self.CosAngle(lmList[24],lmList[26],lmList[28]))
        JointAngleCos.append(self.CosAngle(lmList[23],lmList[25],lmList[27]))
        
        return JointAngleCos
