#!/usr/bin/env python
import os
import cv2
import numpy as np
import keras

class LoadData():

    def __init__(self,emotion_folder,pic_folder):
        self.emotion_folder = emotion_folder
        self.pic_folder = pic_folder


    def read_data(self):
        all_emotions = []
        for root, dirs, files in os.walk(self.emotion_folder):
            for name in files:
                all_emotions.append([os.path.join(root, name),name])

        total_files = len(all_emotions)
        input_files = [None] * total_files
        labels = [None] * total_files
        for index,emotions in enumerate(all_emotions):
            folder = emotions[1].split('_');
            pic_path = os.path.join(self.pic_folder,folder[0],folder[1])
            pic_name = folder[0] + '_' + folder[1] + '_' + folder[2] + '.png'
            picture = os.path.join(pic_path,pic_name)
            input_files[index] = picture
            with open(emotions[0]) as f:
                first_line = f.readline()
            labels[index] = int(float(first_line))

        return input_files,labels

    def preprocess(self,images,labels,img_rows,img_cols,train_percentage,num_classes):
        x = np.zeros((len(images),img_rows,img_cols))
        for index,image_file in enumerate(images):
            img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img,(img_cols,img_rows),interpolation= cv2.INTER_CUBIC)
            x[index] = res

        indices = np.random.permutation(x.shape[0])
        separation = int(train_percentage*x.shape[0])
        training_idx, test_idx = indices[:separation], indices[separation:]
        x_train, x_test = x[training_idx, :], x[test_idx, :]

        labels = np.array(labels)
        y_train,y_test = labels[training_idx], labels[test_idx]

        x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
        x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

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
    img_rows = 480
    img_cols = 640

    loadData = LoadData(emotion_folder,pic_folder)
    image_names,labels = loadData.read_data()
    loadData.preprocess(image_names,labels,img_rows,img_cols,train_ratio,num_of_classes)
