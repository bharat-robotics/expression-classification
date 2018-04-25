#!/usr/bin/env python
from __future__ import print_function

from data import LoadData

import os

import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Merge,Input
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,TensorBoard
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def create_model(x_train,num_of_classes):

    input = Input(shape=x_train.shape[1:])

    model = Sequential()

    conv1_7x7_s2 = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu', name='conv1/7x7_s2',
                                 kernel_regularizer=l2(0.0002))(input)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='same', name='pool1/3x3_s2')(
        conv1_7x7_s2)
    conv2_3x3_reduce = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu', name='conv2/3x3_reduce',
                                      kernel_regularizer=l2(0.0002))(pool1_3x3_s2)
    conv2_3x3 = Conv2D(192, kernel_size=(3,3), padding='same', activation='relu', name='conv2/3x3',
                              kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2/3x3_s2')(
        conv2_3x3)



    inception_3a_1x1 = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu', name='inception_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Conv2D(96, kernel_size=(1,1), padding='same', activation='relu',
                                            name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3 = Conv2D(128,kernel_size=(3, 3), padding='same', activation='relu', name='inception_3a/3x3',
                                     kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
    inception_3a_5x5_reduce = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu',
                                            name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5 = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', name='inception_3a/5x5',
                                     kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
    inception_3a_pool = MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='same', name='inception_3a/pool')(
        pool2_3x3_s2)
    inception_3a_pool_proj = Conv2D(32, kernel_size= (1, 1),padding='same', activation='relu',
                                           name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
    inception_3a_output = Merge(mode='concat',name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])



    inception_3b_1x1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                                     kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3 = Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu', name='inception_3b/3x3',
                                     kernel_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
    inception_3b_5x5_reduce = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5 = Conv2D(96, kernel_size=(5, 5), padding='same', activation='relu', name='inception_3b/5x5',
                                     kernel_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool')(
        inception_3a_output)
    inception_3b_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                           name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_output = Merge(mode='concat', name='inception_3b/output')([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])



    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3/3x3_s2')(inception_3b_output)
    inception_4a_1x1 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4a/1x1',
                                     kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Conv2D(96, kernel_size=( 1, 1), padding='same', activation='relu',
                                            name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3 = Conv2D(208, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4a/3x3',
                                     kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
    inception_4a_5x5_reduce = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5 = Conv2D(48, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4a/5x5',
                                     kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a/pool')( pool3_3x3_s2)
    inception_4a_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                           name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
    inception_4a_output = Merge(mode='concat', name='inception_4a/output')([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj])


    inception_4b_1x1 = Conv2D(160, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4b/1x1',
                                     kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Conv2D(112, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3 = Conv2D(224, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4b/3x3',
                                     kernel_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    inception_4b_5x5_reduce = Conv2D(24, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4b/5x5',
                                     kernel_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b/pool')(inception_4a_output)

    inception_4b_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                           name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = Merge(mode='concat',name='inception_4b_output')([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj])




    inception_4c_1x1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4c/1x1',
                                     kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4c/3x3',
                                     kernel_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    inception_4c_5x5_reduce = Conv2D(24, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4c/5x5',
                                     kernel_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c/pool')(
        inception_4b_output)
    inception_4c_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                           name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = Merge(mode='concat', name='inception_4c/output')([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj])




    inception_4d_1x1 = Conv2D(112, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4d/1x1',
                                     kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Conv2D(144, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3 = Conv2D(288, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4d/3x3',
                                     kernel_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    inception_4d_5x5_reduce = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                            name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4d/5x5',
                                     kernel_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool')(
        inception_4c_output)
    inception_4d_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                           name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = Merge(mode='concat',name='inception_4d/output')([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj])


    loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)
    loss2_conv = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='loss2/conv',
                               kernel_regularizer=l2(0.0002))(loss2_ave_pool)
    conv_model = Model(input=input,output=loss2_conv)


    model.add(conv_model)
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_regularizer=l2(0.0002)))
    model.add(Dropout(0.7))
    model.add(Dense(num_of_classes,activation='softmax'))

    return model

def deeplearning(x_train,y_train,x_test,y_test,num_of_classes,batch_size,epochs):

    model = create_model(x_train, num_of_classes)

    SVG(model_to_dot(model).create(prog='dot', format='svg'))
    model.summary()

    sgd = keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=0.0001, nesterov=False)

    model.compile(loss=['categorical_crossentropy'],
                  optimizer=sgd,
                  metrics=['accuracy'])

    os.system('rm -rf Graph')
    visualize = TensorBoard(log_dir='./Graph', histogram_freq=1,
                                            write_graph=True, write_images=True)


    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,
        vertical_flip=False)


    datagen.fit(x_train)
    dataset = datagen.flow(x_train,y_train,batch_size=batch_size)

    model.fit_generator(dataset,
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[visualize])

    model.evaluate(x_test, y_test, verbose=2)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__== "__main__":
    emotion_folder = '/home/cokcybot/Expression-Classification/Emotion'
    pic_folder = '/home/cokcybot/Expression-Classification/cohn-kanade-images'

    train_ratio = 0.8
    num_of_classes = 8

    img_rows = 224
    img_cols = 224
    channels = 1

    batch_size = 100
    epochs = 200

    loadData = LoadData(emotion_folder,pic_folder)
    train_images, train_labels, test_images, test_labels = loadData.read_data(train_ratio)

    x_train, x_test, y_train, y_test = loadData.preprocess(train_images, train_labels, test_images, test_labels,img_rows,img_cols,train_ratio,num_of_classes)


    x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,channels)
    x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,channels)

    deeplearning(x_train,y_train,x_test,y_test,num_of_classes,batch_size,epochs)

