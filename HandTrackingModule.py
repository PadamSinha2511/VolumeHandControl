import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self,mode=False,maxHands=2,minDetect=0.5, minTrack=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.minDetect=minDetect
        self.minTrack=minTrack

        self.mpHand = mp.solutions.hands
        self.hand = self.mpHand.Hands(self.mode,self.maxHands,self.minDetect,self.minTrack)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingerTips=[4,8,12,16,20]

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
               if draw:
                   self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)


        return img

    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        #xmin , xmax=0,0
        #ymin , ymax=0,0
        bbox=[]
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id,cx,cy])
                # if id==11
                # if draw:
                #     cv2.circle(img,(cx,cy),15,(255,0,0),2,cv2.FILLED)
            xmin , xmax = min(xList), max(xList)
            ymin , ymax = min(yList), max(yList)
            bbox=xmin,ymin,xmax,ymax
            if draw:
                cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)
        return self.lmList,bbox

    def fingersUp(self):
        fingers = []
        #Thumb
        if self.lmList[self.fingerTips[0]][1] > self.lmList[self.fingerTips[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 Fingers
        for id in range(1, 5):
            if self.lmList[self.fingerTips[id]][2] < self.lmList[self.fingerTips[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self,p1,p2,img,draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length,img,[x1,y1,x2,y2]

def main():
    cTime = 0
    pTime = 0
    cap=cv2.VideoCapture(0)
    detector=HandDetector()
    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)

        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("Output", img)
        key=cv2.waitKey(1)
        if key==81 or key==113:
            break

if __name__=="__main__":
    main()