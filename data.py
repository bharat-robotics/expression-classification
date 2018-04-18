#!/usr/bin/env python
import os
from shutil import copytree, copy2
import copy

class Data():

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
            first_line=[]
            with open(emotions[0]) as f:
                first_line = f.readline()
            labels[index] = int(float(first_line))

        print(input_files)
        print(labels)


if __name__== "__main__":
    emotion_folder = '/home/bharat/Expression-Classification/Emotion'
    pic_folder = '/home/bharat/Expression-Classification/cohn-kanade-images'
    loadData = Data(emotion_folder,pic_folder)
    loadData.read_data()
