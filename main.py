import cv2
import numpy as np
import keyboard

file = input("""Please enter the filename and extension you want to analyze.\n
Note: The file needs to be in the same directory as the program.\n """)

# Motion detection using contours

video = cv2.VideoCapture(file)

ret, frame1 = video.read()
ret, frame2 = video.read()

while video.isOpened():
    difference = cv2.absdiff(frame1, frame2)
    grayscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 950:
            continue
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 1)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (120, 0, 150), 2)

    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = video.read()

    if cv2.waitKey(50) == 27:
        break

    if keyboard.is_pressed('q'):
        break



cv2.destroyAllWindows()
video.release()
