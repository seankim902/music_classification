# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from models import RNN

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Activation, Dropout

from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split


    
    
def main():


    y_class = 10
    dim_freq = 11025
    dim = 512
    dim_ctx = 4096
    
    os.chdir('/home/seonhoon/Desktop/workspace/music/data/')
    train=pd.read_pickle('train.pkl')
    valid=pd.read_pickle('valid.pkl')
    
    train_x=np.array([ q for q in train['x'] ]).reshape(-1,56,dim_freq).astype('float32')
    train_y=[ a for a in train['y'] ]
    train_y=np.array(train_y)[:,None]
    train_y = np_utils.to_categorical(train_y, y_class).astype('int32')
    train_img = np.array([ f for f in train['cnn_feature'] ]).astype('float32')
    
    
    valid_x=np.array([ q for q in valid['x'] ]).reshape(-1,56,dim_freq).astype('float32')
    valid_y=[ a for a in valid['y'] ]
    valid_y=np.array(valid_y)[:,None]
    valid_y_original=valid_y
    valid_y = np_utils.to_categorical(valid_y, y_class).astype('int32')
    valid_img = np.array([ f for f in valid['cnn_feature'] ]).astype('float32')



    '''
    model = Sequential()
    model.add(TimeDistributedDense( dim_freq, hidden))
    model.add(Dropout(0.2))

    model.add(LSTM( hidden, hidden, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM( hidden, hidden))
    model.add(Dropout(0.4))

    model.add(Dense(hidden, y_class))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop) 
  #  model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode='categorical')
    model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(valid_x, valid_y))
    '''

    print 'train x :', train_x.shape
    print 'train y : ', train_y.shape
    print 'train imgs :', train_img.shape
    
    print 'valid x :', valid_x.shape
    print 'valid y : ', valid_y.shape
    print 'valid imgs :', valid_img.shape
    


    
    model = RNN(y_class, dim_freq, dim, dim_ctx)

    model.train(train_x, train_img, train_y, 
                valid_x, valid_img, valid_y_original, valid=1,
                lr=0.0002, dropout=0.999, batch_size=256, epoch=100, save=10)
if __name__ == '__main__':
    main() 
    
    