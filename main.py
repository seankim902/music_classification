# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from models import RNN_GRU, BIRNN_GRU, BIRNN_LSTM, BIRNN_GRU2

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Activation, Dropout

from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split
from draw_cm import *

    
    
def main():


    y_class = 10
    dim_freq = 5512
    dim = 1024
    step=112
    dim_ctx = 4096
    
    os.chdir('/home/seonhoon/Desktop/workspace/music/data6/')
    train=pd.read_pickle('train_fft.pkl')
    valid=pd.read_pickle('valid_fft.pkl')
    train_x=np.array([ q for q in train['x'] ]).reshape(-1,step,dim_freq).astype('float32')
    train_y=[ a for a in train['y'] ]
    train_y=np.array(train_y)[:,None]
    train_y = np_utils.to_categorical(train_y, y_class).astype('int32')
    train_img = np.array([ f for f in train['cnn_feature'] ]).astype('float32')

    
    valid_x=np.array([ q for q in valid['x'] ]).reshape(-1,step,dim_freq).astype('float32')
    valid_y=[ a for a in valid['y'] ]
    valid_y_true = valid_y
    valid_y=np.array(valid_y)[:,None]
    valid_y_original=valid_y
    valid_y = np_utils.to_categorical(valid_y, y_class).astype('int32')
    valid_img = np.array([ f for f in valid['cnn_feature'] ]).astype('float32')

    print 'train x :', train_x.shape
    print 'train y : ', train_y.shape
    print 'train imgs :', train_img.shape

    print 'valid x :', valid_x.shape
    print 'valid y : ', valid_y.shape
    print 'valid imgs :', valid_img.shape
    


    
    model = BIRNN_GRU(y_class, dim_freq, dim, dim_ctx)
 
    model.train(train_x, train_img, train_y, 
                valid_x, valid_img, valid_y_original, valid=1,
                lr=0.0001, dropout=0.8, batch_size=512, epoch=100, save=10)

    prediction, probs = model.prediction(valid_x, valid_img, valid_y, lr=0.0001, batch_size=512)

    print valid_y_true
    print prediction
    correct = 0 
    for i in xrange(len(valid_y_true)):
        if valid_y_true[i]==prediction[i]:
            correct += 1
    print correct
    
    true = pd.DataFrame(valid_y_true) 
    pred = pd.DataFrame(prediction) 
    true.to_pickle('/home/seonhoon/Desktop/workspace/music/data6/true.pkl')
    pred.to_pickle('/home/seonhoon/Desktop/workspace/music/data6/pred.pkl')
    draw_cm(valid_y_true, prediction)

if __name__ == '__main__':
    main() 
    
    