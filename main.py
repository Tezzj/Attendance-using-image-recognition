# install these packages
# pip install cmake
# pip install opencv-python
# pip install cc
import csv

import numpy as np
import cv2
from datetime import datetime
import face_recognition

# Make a new variable to capture the video
video_capture = cv2.VideoCapture(0)

# Load known faces
My_image = face_recognition.load_image_file('images/Me.jpg')
My_encoding = face_recognition.face_encodings(My_image)[0]

other_image = face_recognition.load_image_file('images/other.jpg')
other_encoding = face_recognition.face_encodings(other_image)[0]

known_face_encoding = [My_encoding, other_encoding]
known_face_names = ['Tejas', 'Sallu']

# List of expected people
people = known_face_names.copy()
face_locations = []
face_encoding = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime('%d-%m-%Y')

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


# Recognise faces
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)

for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
    face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
    best_match_index = np.argmin(face_distance)

    if matches([best_match_index]):
        name = known_face_names[best_match_index]

    # Add the text if a person is present
    if name in known_face_names:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 1.5
        fontColor = (255, 0, 0)
        thickness = 3
        lineType = 2
        cv2.putText(frame, name + 'Present', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    if name in students:
        students.remove(name)
        current_time = now.strftime('%H-%M%S')
        lnwriter.writerow([name, current_time])

    cv2.imshow('Attendance', frame)
    if cv2.waitkey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()