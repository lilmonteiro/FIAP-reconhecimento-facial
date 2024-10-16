import cv2
import os
import numpy as np
import face_recognition

def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder):
        if (filename.endswith('jpeg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                face_encoding = face_encodings[0]
                name = os.path.splitext(filename)[0][:-1]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
    
    return known_face_encodings, known_face_names

def main():
    image_folder = 'images'
    known_face_encodings, known_face_names = load_images_from_folder(image_folder)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:,:,::-1])
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces ( known_face_encodings, face_encoding)
            name = 'Desconhecido'
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name) 
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *=4
            right *=4
            bottom *=4
            left *=4

            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
            cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom -6), font, 1.0, (255,255,255), 1)

        cv2.imshow('Face detection', frame)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        