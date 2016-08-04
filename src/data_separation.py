import csv
import os
import numpy as np
import cv2

write_file_train = '/home/marta/PycharmProjects/learning_caffe/resource/for_train.txt'
write_file_test = '/home/marta/PycharmProjects/learning_caffe/resource/for_test.txt'

poin_train = open(write_file_train, 'w')
poin_test = open(write_file_test, 'w')

base_path_train = '/home/marta/PycharmProjects/learning_caffe/resource/train/'
base_path_test = '/home/marta/PycharmProjects/learning_caffe/resource/test/'


i = 0
with open('/home/marta/PycharmProjects/learning_caffe/resource/fer2013.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        i += 1
        if i == 1:
            continue
        cl = row[0]
        img = row[1]
        us = row[-1]
        p = np.fromstring(img, dtype=np.uint8, sep=' ')
        a = p.reshape(48, 48)
        if us == "Training":
            path = base_path_train + cl
            if not os.path.exists(path):
                os.makedirs(path)
            path = path + "/" + str(i) + ".png"
            try:
                cv2.imwrite(path, a)
                path = path + " " + cl + "\n"
                poin_train.write(path)
            except Exception:
                print "Error with picture " + str(i)
        else:
            path = base_path_test + cl
            if not os.path.exists(path):
                os.makedirs(path)
            path = path + "/" + str(i) + ".png"
            try:
                cv2.imwrite(path, a)
                path = path + " " + cl + "\n"
                poin_test.write(path)
            except Exception:
                print "Error with picture " + str(i)

poin_train.close()
poin_test.close()