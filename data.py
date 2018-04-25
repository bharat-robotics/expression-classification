#!/usr/bin/env python
import os
import cv2
import numpy as np
import keras
from random import shuffle
class LoadData():

    def __init__(self,emotion_folder,pic_folder):
        self.emotion_folder = emotion_folder
        self.pic_folder = pic_folder


    def read_data(self,train_percentage):
        all_emotions = []
        for root, dirs, files in os.walk(self.emotion_folder):
            for name in files:
                all_emotions.append([os.path.join(root, name),name])

        total_files = len(all_emotions)
        input_files = [None] * total_files
        for index,emotions in enumerate(all_emotions):
            folder = emotions[1].split('_')
            pic_path = os.path.join(self.pic_folder,folder[0],folder[1])
            pic_name = folder[0] + '_' + folder[1] + '_' + folder[2] + '.png'
            picture = os.path.join(pic_path,pic_name)
            with open(emotions[0]) as f:
                first_line = f.readline()
            input_files[index] = [picture,int(float(first_line))]

        shuffle(input_files)
        separation = int(train_percentage*len(input_files))
        image_train =  input_files[:separation]
        image_test = input_files[separation:]
        shuffle(image_test)

        datas  = []
        for index,data in enumerate(image_train):
            file = data[0]
            splits = file.split('_')
            pic_seq = int(splits[2].split('.')[0])
            for i in range(pic_seq,pic_seq-5,-1):
                if(i < 10):
                    pic_string = '0000000' + str(i)
                else:
                    pic_string = '000000' + str(i)
                pic_name = splits[0] + '_' + splits[1] + '_' + pic_string + '.png'
                datas.append([pic_name,data[1]])

        shuffle(datas)

        training_images = [None] * len(datas)
        training_labels = [None] * len(datas)
        testing_images = [None] * len(image_test)
        testing_labels = [None] * len(image_test)

        for index,image_file in enumerate(datas):
            training_images[index] = image_file[0]
            training_labels[index] = image_file[1]

        for index,image_file in enumerate(image_test):
            testing_images[index] = image_file[0]
            testing_labels[index] = image_file[1]

        return training_images,training_labels,testing_images,testing_labels

    def preprocess(self,train_images,train_labels,test_images,test_labels,img_rows,img_cols,train_percentage,num_classes):
        x_train = np.zeros((len(train_images),img_rows,img_cols))
        y_train = np.zeros(len(train_labels))
        x_test = np.zeros((len(test_images), img_rows, img_cols))
        y_test = np.zeros(len(test_images))

        for index,image_file in enumerate(train_images):
            img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img,(img_cols,img_rows),interpolation= cv2.INTER_CUBIC)
            x_train[index] = res
            y_train[index] = train_labels[index]

        for index, image_file in enumerate(test_images):
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
            x_test[index] = res
            y_test[index] = test_labels[index]

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = np.array(keras.utils.to_categorical(y_train, num_classes))
        y_test = np.array(keras.utils.to_categorical(y_test, num_classes))

        return x_train,x_test,y_train,y_test

if __name__== "__main__":
    emotion_folder = '/home/cokcybot/Expression-Classification/Emotion'
    pic_folder = '/home/cokcybot/Expression-Classification/cohn-kanade-images'

    train_ratio = 0.8
    num_of_classes = 8
    img_rows = 224
    img_cols = 224

    loadData = LoadData(emotion_folder,pic_folder)

    train_images,train_labels,test_images,test_labels = loadData.read_data(train_ratio)


    x_train,x_test,y_train,y_test = loadData.preprocess(train_images,train_labels,test_images,test_labels,img_rows,img_cols,train_ratio,num_of_classes)

    x_test *= 255.0
    x_test = x_test.astype('uint8')
    for t in range(x_test.shape[0]):
        img = x_test[t,:]
        cv2.imshow('img',img)
        cv2.waitKey(0)
        print(y_test[t])