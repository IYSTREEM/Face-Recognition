3333333333
import face_recognition as fr
import cv2
import numpy as np
import os

path = "./train/"

known_names = []
known_name_encodings = []

images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

print(known_names)

video_path = "./test/hi.mp4"
video_capture = cv2.VideoCapture(video_path)

# Set the playback speed and frames to skip 
playback_speed = 0.0 
frames_to_skip =  1
video_capture.set(cv2.CAP_PROP_FPS, video_capture.get(cv2.CAP_PROP_FPS) * playback_speed)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')    
output_video_path = "./output.avi"
output_video = cv2.VideoWriter(output_video_path, fourcc, video_capture.get(cv2.CAP_PROP_FPS), (360, 240))

frame_counter = 0
while True:
    ret, frame = video_capture.read()

    if not ret:
        break # Break the loop if the video ends

    if frame_counter % frames_to_skip != 0:
        frame_counter += 1
        continue 

    # Resize the frame to 360x240 resolution
    frame = cv2.resize(frame, (360, 240))

    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    output_video.write(frame)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # Break the loop if 'q' is pressed

    frame_counter += 1

# Release the video capture object, VideoWriter object, and close all windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()