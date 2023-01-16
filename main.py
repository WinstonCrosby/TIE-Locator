import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui as pg
import time
import os
import random


if __name__ == '__main__':

    # n = 0
    # t = 5
    #
    # f = open('pos.txt', 'w')
    #
    # while n<1500:
    #     img = pg.screenshot()
    #     img_np = np.array(img)
    #     # cv2.imwrite("img_np.png", img_np)
    #     # img_np = cv2.imread(img_np.png)
    #     frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     # frame = frame[475:600, 1375:1500]
    #     randint1 = random.randint(0, 1700)
    #     randint2 = random.randint(randint1, 1920)
    #     randint3 = random.randint(0, 800)
    #     randint4 = random.randint(randint3, 1080)
    #     frame = frame[randint3:randint4, randint1:randint2]
    #
    #     # cv2.imshow('frame', frame)
    #     # cv2.waitKey(0)  # Keeps image open indefinitely
    #     # cv2.destroyAllWindows()  # Destroys all windows
    #
    #     r = random.randint(0, 5000000)
    #
    #     filename = 'C:/Users/Winston/PycharmProjects/CS549_Project/neg/file_%dsn.png'%r
    #     cv2.imwrite(filename, frame)
    #
    #     n = n+1
    #
    #
    #     time.sleep(t)
    #     t = 0.1
    #
    # f.close()

    # with open('neg.txt', 'w') as f:
    #     for filename in os.listdir('neg'):
    #         f.write('neg/' + filename + '\n')

    # with open('pos.txt', 'w') as f:
    #     for filename in os.listdir('pos'):
    #         f.write('pos/' + filename + '\t' + '1' + '\t' + '0 0 125 125' + '\n')

    cascade_TIE = cv2.CascadeClassifier('cascade/cascade.xml')

    while True:

        img = pg.screenshot()
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        rectangles = cascade_TIE.detectMultiScale(frame)

        for (x,y,w,h) in rectangles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        cv2.imshow('Matches', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()  # Destroys all windows


