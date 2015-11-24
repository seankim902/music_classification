# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# import some data to play with


genre={'reggae':0,'jazz':1,'metal':2,'hiphop':3,'blues':4,'classical':5,'country':6,'pop':7,'disco':8,'rock':9}

keys=[]
for key, value in sorted(genre.iteritems(), key=lambda (k,v): (v,k)):
    keys.append(key)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, keys, rotation=45)
    plt.yticks(tick_marks, keys)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def draw_cm(y_test, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=1)

    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
 
    plt.show()