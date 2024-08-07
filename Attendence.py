import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime   
imgs_path = "Images"
images = []
names = []

imgs_list = os.listdir(imgs_path)
for cls_name in imgs_list:
    img_path = os.path.join(imgs_path,cls_name)
    current_img = cv2.imread(img_path)
    images.append(current_img)
    names.append(cls_name.split(".")[0])


def mark_attendence(name):
    with open("Attendence.csv",'r+') as f:
        cache = f.readlines()
        name_list = []
        for line in cache:
            entry = line.split(',')
            name_list.append(entry[0])
        
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dt_string}")
def get_images_encoding(images):
    encoding_list = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list

encoding_list = get_images_encoding(images)
print("Encoding Complete!")
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    if not ret:
        break
    
    frame_scaled = cv2.resize(frame,(0,0),None,0.25,0.25) # Scale = .25
    frame_rgb = cv2.cvtColor(frame_scaled,cv2.COLOR_BGR2RGB)
    faces_loc = face_recognition.face_locations(frame_rgb)
    encodings = face_recognition.face_encodings(frame_rgb,faces_loc)

    for encode,loc in zip(encodings,faces_loc):
        matches = face_recognition.compare_faces(encoding_list,encode)
        face_dist = face_recognition.face_distance(encoding_list,encode)
        match_index = np.argmin(face_dist)
        if matches[match_index]:
            name = names[match_index]
            mark_attendence(name)
            y1,x2,y2,x1 = list(map(lambda x:x*4,loc))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2+40),(x2,y2),(0,255,0),-1)
            cv2.putText(frame,name,(x1+6,y2+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
            

    
    
    cv2.imshow("Camera",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
