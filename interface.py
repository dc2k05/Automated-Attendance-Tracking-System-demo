import cv2
import numpy as np
import os
from datetime import datetime

# Load the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to train the recognizer using images
def train_recognizer(training_data_dir):
    face_samples = []
    ids = []
    label_names = {}
    
    # Loop through each person's folder and read their images
    for idx, person in enumerate(os.listdir(training_data_dir)):
        label_names[idx] = person
        person_folder = os.path.join(training_data_dir, person)
        
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y+h, x:x+w])
                ids.append(idx)
    
    # Train the recognizer
    recognizer.train(face_samples, np.array(ids))
    return label_names

# Mark attendance by writing to a file
def mark_attendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        date_string = now.strftime('%Y-%m-%d')
        f.write(f'{name},{date_string},{time_string}\n')

# Main attendance system
def attendance_system(training_data_dir):
    # Train recognizer and get label names
    label_names = train_recognizer(training_data_dir)
    
    # Start webcam
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            
            if confidence < 100:
                name = label_names[label]
                mark_attendance(name)
            else:
                name = "Unknown"
            
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Display the video feed
        cv2.imshow('Video', frame)
        
        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the system (replace with the directory containing known face images)
attendance_system('training_data')