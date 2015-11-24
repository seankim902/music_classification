# -*- coding: utf-8 -*-

import numpy as np
from cnn import *
import os
import pandas as pd




os.chdir('/home/seonhoon/Desktop/workspace/music/data6/')
data=pd.read_pickle('valid_fft.pkl')

files= data['file']
nfiles=[]

for i in range(len(files)):
    nfiles.append(files[i].replace('data6','data6/specgram').replace('wav','jpg'))






cnn = CNN(deploy='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt', model='/home/seonhoon/Desktop/caffemodel/vgg19/VGG_ILSVRC_19_layers.caffemodel')
featurelist = cnn.get_features(nfiles, layer='fc7')

data['cnn_feature']=0
data['cnn_feature']=data['cnn_feature'].astype(object)

data['jpg_file']=0
data['jpg_file']=data['jpg_file'].astype(object)

for j in xrange(len(featurelist)):
    print j
    data.set_value(j,'cnn_feature',featurelist[j])
    data.set_value(j,'jpg_file',nfiles[j].split('/')[-1])

data.to_pickle('valid_fft.pkl')
