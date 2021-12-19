#Resources
#https://github.com/BakingBrains/Pose_estimation/blob/main/PoseModule.py
#https://www.youtube.com/watch?v=5kaX3ta398w
#https://github.com/BakingBrains/Pose_estimation/blob/main/pose_estimation.py
#https://www.tensorflow.org/lite/examples/pose_estimation/overview

import cv2
import mediapipe as mp
import time
import math
import numpy as np

class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = False, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        self.lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y *h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        #Calculate the Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2) )
        if angle < 0:
            angle += 360
        
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            
            #cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) 

        return angle
    
def main():
    #Create a VideoCapture object and read from input file
    #If the input is the camara, pass 0 instead of the video file name
    cap = cv2.VideoCapture(0)  #'./a.mp4'make VideoCapture(0) for webcam
    pTime = 0
    detector = PoseDetector()
   
    #Check if the camara opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    per = 0
    dir = 0
    count = 0
    
    #Read Until Video is Completed
    while cap.isOpened():
        #Capture frame by frame
        success, img = cap.read()
        img = cv2.resize(img, (1020, 720))
        img = detector.findPose(img, True)
        lmList = detector.findPosition(img, True)
        #print(lmList)


        if success == True:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            if len(lmList) !=0:
                # Right Arm
                angle = detector.findAngle(img, 11, 13, 15 )
                per = np.interp(angle, (242, 290), (0, 100))
                bar = np.interp(angle, (242, 290), (620, 100))
                #print(angle, per)
                # Left Arm
                angle2 = detector.findAngle(img, 12, 14, 16 )
                per2 = np.interp(angle2, (242, 290), (0, 100))
                bar2 = np.interp(angle2, (242, 290), (620, 100))
                #Check for the dumbbell curls
                color = (255, 0, 255)
                if (per) == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count+=0.5   
                        dir = 1
                               
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count+=0.5
                        dir = 0
                #print(count)
                        
                #Display the resulting frame
                #Draw Progress Bar
                cv2.rectangle(img, (1100, int(bar)), (900, 720), color, cv2.FILLED)
                cv2.putText(img, f'{str(int(per))}%', (890, 720), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)


                #Draw Curl Count
                cv2.rectangle(img, (0, 490), (230, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)
                cv2.imshow("Trainner", img)
                
            #Press Q on Keyboard to Exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        #Break the loop
        else:
            break
        
    #When everythoing done, release the video capture objetc
    cap.release()
    #Close all the frames
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
  main() 