import cv2
import os

alphabet_folder_train = "..\\sign_language\\dados-libras\\train"
alphabet_folder_test = "..\\sign_language\\dados-libras\\test"

alphabet = ['H', 'J', 'K', 'X', 'Z']

frameRate = 0.3

for letter in alphabet:
    output_folder = os.path.join(alphabet_folder_test, letter)
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(1, 26):
        video_path = os.path.join(alphabet_folder_test, letter, str(i) + ".mp4")
        vidcap = cv2.VideoCapture(video_path)
        
        sec = 0
        count = 1
        
        while True:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            hasFrames, image = vidcap.read()
            
            if not hasFrames:
                break
            
            frame_filename = os.path.join(output_folder, f"{letter}{count}_{i}.jpg")
            cv2.imwrite(frame_filename, image)
            
            sec += frameRate
            sec = round(sec, 2)
            count += 1
