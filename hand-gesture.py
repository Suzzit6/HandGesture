import cv2
import mediapipe as mp
import math
import numpy as np
from setvolume import setVolume

def getThumb(results,frame):
    thumb_tip_x, thumb_tip_y = None, None
    if results.multi_hand_landmarks:
        for hand_landmark,hand_handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            if hand_handedness.classification[0].label == "Right":
              thumb_tip = hand_landmark.landmark[4]
            #   print(thumb_tip)
              thumb_tip_x, thumb_tip_y = int(thumb_tip.x*frame.shape[1]), int(thumb_tip.y*frame.shape[0])
              cv2.circle(frame,(thumb_tip_x,thumb_tip_y),5,(0,0,255),-1)
    return thumb_tip_x, thumb_tip_y 
            
def getIndex(results,frame):
    index_tip_x, index_tip_y = None, None
    if results.multi_hand_landmarks:
        for hand_landmark,hand_handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            if hand_handedness.classification[0].label == "Right":
             index_tip = hand_landmark.landmark[8]
             # print(index_tip)
             index_tip_x, index_tip_y = int(index_tip.x*frame.shape[1]), int(index_tip.y*frame.shape[0])
             cv2.circle(frame,(index_tip_x,index_tip_y),5,(0,0,255),-1)            
    return index_tip_x,index_tip_y        

def getDistance(results,frame):
    distance = None
    thumb_tip_x,thumb_tip_y = getThumb(results,frame)
    index_tip_x,index_tip_y = getIndex(results,frame)
    
    if thumb_tip_x is not None and thumb_tip_y is not None and index_tip_x is not None and index_tip_y is not None:
     distance = math.sqrt((index_tip_x - thumb_tip_x)**2 + (index_tip_y - thumb_tip_y)**2)
    print(f"distance between thumb and index ", distance)
    return distance 


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.7,min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    # get thumb tip
            
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmark,mp_hands.HAND_CONNECTIONS)
    distance = getDistance(results=results,frame=frame)
    volume = None
    if distance is not None:
       volume = np.interp(distance,[50,300],[0.0,1.0])
       setVolume(volume=volume)
    
       cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 2)  # Volume bar outline
       volume_bar = int(np.interp(volume, [0.0, 1.0], [400, 150]))
       cv2.rectangle(frame, (50, volume_bar), (85, 400), (255, 0, 0), -1)  # Current volume
       cv2.putText(frame, f'Volume: {int(volume * 100)}%', (40, 450),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
    cv2.imshow("HAND TRACKS", frame)        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    