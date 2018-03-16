from sklearn.ensemble import RandomForestClassifier
from imutils import resize
import pandas as pd
import numpy as np
import time
import cv2
import json

dataX  = pd.read_csv('./dataset/x_values_1.csv')
y = pd.read_csv('./dataset/y_values_1.csv')

dict_cont = {'Jet Azul' : 0, 'Jumbo Flow Negra' : 0, 'Jumbo Flow Blanca' : 0, 'Jumbo Mix Naranja' : 0, 'Jumbo Roja' : 0, 
             'Chocorramo' : 0, 'Fruna Verde' : 0, 'Fruna Naranja' : 0, 'Fruna Roja' : 0, 'Fruna Amarilla' : 0}

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(dataX, np.ravel(y))


def get_exact_name(list_with_number):
    number = list_with_number[0]
    key = ''
    if number == 0:
        key = 'Jet Azul'
    if number == 1:
        key = 'Jumbo Flow Negra'
    if number == 2:
        key = 'Jumbo Flow Blanca'
    if number == 3:
        key = 'Jumbo Mix Naranja'
    if number == 4:
        key = 'Jumbo Roja'
    if number == 5:
        key = 'Chocorramo'
    if number == 6:
        key = 'Fruna Verde'
    if number == 7:
        key = 'Fruna Naranja'
    if number == 8: 
        key = 'Fruna Roja'
    if number == 9:
        key = 'Fruna Amarrilla'
    
    dict_cont[key] = dict_cont[key] + 1
    with open('./output/out.txt', 'w') as outputfile:
        json.dump(dict_cont, outputfile, sort_keys=True)
    return key


def getRGB(kernel_op, kernel_cl, image):
    img1 = cv2.imread("dataset/banda.jpeg")
    img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel_op, iterations=2)
    img2 = image
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel_cl, iterations=2)
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


def process_each_frame(kernel_op, kernel_cl, back_ground_substractor, cap, found):
    ret, frame = cap.read() 
    frame = frame[::, 95:525]
    image = frame
    image = resize(image, width=500)
    image = image[50:3500, 75:480]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = back_ground_substractor.apply(image)
    cv2.imshow('No Background', thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_op, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_cl, iterations=2)
    im, contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0 and cv2.contourArea(contours[0]) > 10000 and cv2.contourArea(contours[0]) < 80000:
        if found != True:
            found = True
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(0,0,255),2)
            if rect[0][1] > 290 and rect[0][1] < 325:
                area = rect[1][0] * rect[1][1]
                rgb  = getRGB(kernel_op, kernel_cl, frame)
                print('Area: ', area)
                print('Color: ', rgb)
                data = rgb + [area]
                data_predicted = clf.predict([data])
                name = get_exact_name(data_predicted)
                print(name)
    else:
        found = False
    cv2.imshow("objects Found", image)
    cv2.imshow('Thresh', thresh)
    time.sleep(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


def get_frames_and_predict_type(): 
    # Variable declarations 
    kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    back_ground_substractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    found = False
    cap = cv2.VideoCapture(0)
    
    # Camera settings
    cap.set(3,640)
    cap.set(4,480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE , 0.4)

    # Process each frame
    while(True):
        process_each_frame(kernel_op, kernel_cl, back_ground_substractor, cap, found)        
    
    # Realease the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_frames_and_predict_type()
