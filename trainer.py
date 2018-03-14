import cv2
import numpy as np
import time 
from imutils import resize
import csv


def write_col_x(rowcita):
    with open('x_values.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(rowcita)

def write_col_y(label):
    with open('y_values.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(label)

def getRGB(image):
    kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    img1 = cv2.imread("videos/banda.jpeg")
    img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernelOP, iterations=2)
    img2 = image
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernelCL, iterations=2)
    diff = cv2.absdiff(img2, img1)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = 35
    imask =  mask>th
    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]
    rprom = 0
    gprom = 0
    bprom = 0
    cont = 0
    a, b, c = canvas.shape
    zero = np.array([0,0,0])
    for i in range(a-1):
        for j in range(b-1):
            arr = canvas[i][j]
            if ((arr > 150).all()):
                bprom += arr[0]
                gprom += arr[1]
                rprom += arr[2]
                cont += 1

    return [int(rprom/cont),int(gprom/cont),int(bprom/cont)]


kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
cap = cv2.VideoCapture('videos/fruna_amarilla_1.avi')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
while(True): 
    ret, frame = cap.read()
    image = frame
    image = resize(image, width=500)
    image = image[50:3500, 75:480]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    thresh = fgbg.apply(image)
    cv2.imshow('No Background', thresh)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOP, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelCL, iterations=2)
    im, contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    if len(contours) > 0 and cv2.contourArea(contours[0]) > 10000 and cv2.contourArea(contours[0]) < 30000:
        area = cv2.contourArea(contours[0])
        rgb  = getRGB(frame)
        print('Area: ', area)
        print('Color: ', rgb)
        write_col_x(rgb + [area])
        write_col_y('8')
        cv2.drawContours(image, contours, -1, (0,255,0), 2)
    cv2.imshow("objects Found", image)
    cv2.imshow('Thresh', thresh)
    time.sleep(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
