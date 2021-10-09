import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###########################
wCam, hCam=640,480
cTime=0
pTime=0
###########################
cap=cv2.VideoCapture(0)
detector=htm.HandDetector(minDetect=0.7,maxHands=1)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]
vol=0
volBar=400
volPercent=0
area=0
colorVol=(255,0,0)
while True:
    success,img=cap.read()
    #Find Hand
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img,draw=True)
    if len(lmList) !=0:

        #Filter based on size
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])//100
        # print(area)
        if 250<area<1000:
            # print("Yes")
            # Find distance between index and thumb
            length,img,info=detector.findDistance(4,8,img)

            #Convert volume

            # print(length)
            # Hand range 40-380
            # Volrange -65 - 0

            volBar = np.interp(length, [50, 200], [400, 150])
            volPercent = np.interp(length, [50, 200], [0, 100])
            # volume.SetMasterVolumeLevel(vol, None)

            #Reduce resolution to make it smoother
            smooth=5
            volPercent=smooth*round(volPercent/smooth)

            #Check fingers up
            fingers=detector.fingersUp()
            print(fingers)
            #Check if pinky is down set volume
            if fingers[4]==0:
                volume.SetMasterVolumeLevelScalar(volPercent / 100, None)
                cv2.circle(img, ((info[0] + info[2]) // 2, (info[1] + info[3]) // 2), 10, (0, 255, 0), cv2.FILLED)
                colorVol=(0,255,0)

            else:
                colorVol=(255,0,0)

    # Drawings
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img,f'{int(volPercent)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cVol=int(volume.GetMasterVolumeLevelScalar()*100)+1
    cv2.putText(img, f'Vol set {int(cVol)}', (350, 50), cv2.FONT_HERSHEY_PLAIN, 3, colorVol, 2)
    # Frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)

    cv2.imshow("Output",img)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break