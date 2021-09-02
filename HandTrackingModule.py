import cv2
import mediapipe as mp
import time


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
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id,cx,cy])
                # if id==11:
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,0),2,cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []

        if self.lmList[self.fingerTips[0]][1] < self.lmList[self.fingerTips[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.fingerTips[id]][2] < self.lmList[self.fingerTips[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

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