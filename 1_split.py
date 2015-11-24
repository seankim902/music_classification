# -*- coding: utf-8 -*-

import os
import random
import shutil
import pandas as pd

##########
count = 10 # valid count
data_path = 'data6'
##########

os.mkdir('/home/seonhoon/Desktop/workspace/music/'+data_path+'/train/')
os.mkdir('/home/seonhoon/Desktop/workspace/music/'+data_path+'/valid/')
os.mkdir('/home/seonhoon/Desktop/workspace/music/'+data_path+'/specgram/')
os.mkdir('/home/seonhoon/Desktop/workspace/music/'+data_path+'/specgram/train')
os.mkdir('/home/seonhoon/Desktop/workspace/music/'+data_path+'/specgram/valid')



path='/home/seonhoon/Desktop/workspace/music/'+data_path+'/wav/'
train_path='/home/seonhoon/Desktop/workspace/music/'+data_path+'/train/'
valid_path='/home/seonhoon/Desktop/workspace/music/'+data_path+'/valid/'

reggae=[]
jazz=[]
metal=[]
hiphop=[]
blues=[]
classical=[]
country=[]
pop=[]
disco=[]
rock=[]

train = []
valid = []

dir_ = os.listdir( path )
for file_ in dir_:
   if 'reggae'==file_.split('.')[0]:
       reggae.append(file_)
   elif 'jazz'==file_.split('.')[0]:
       jazz.append(file_)
   elif 'metal'==file_.split('.')[0]:
       metal.append(file_)
   elif 'hiphop'==file_.split('.')[0]:
       hiphop.append(file_)
   elif 'blues'==file_.split('.')[0]:
       blues.append(file_)
   elif 'classical'==file_.split('.')[0]:
       classical.append(file_)
   elif 'country'==file_.split('.')[0]:
       country.append(file_)
   elif 'pop'==file_.split('.')[0]:
       pop.append(file_)
   elif 'disco'==file_.split('.')[0]:
       disco.append(file_)
   elif 'rock'==file_.split('.')[0]:
       rock.append(file_)

random.shuffle(reggae)
random.shuffle(jazz)
random.shuffle(metal)
random.shuffle(hiphop)
random.shuffle(blues)
random.shuffle(classical)
random.shuffle(country)
random.shuffle(pop)
random.shuffle(disco)
random.shuffle(rock)

random.shuffle(reggae)
random.shuffle(jazz)
random.shuffle(metal)
random.shuffle(hiphop)
random.shuffle(blues)
random.shuffle(classical)
random.shuffle(country)
random.shuffle(pop)
random.shuffle(disco)
random.shuffle(rock)


valid+=(reggae[:count])
valid+=(jazz[:count])
valid+=(metal[:count])
valid+=(hiphop[:count])
valid+=(blues[:count])
valid+=(classical[:count])
valid+=(country[:count])
valid+=(pop[:count])
valid+=(disco[:count])
valid+=(rock[:count])


train+=(reggae[count:])
train+=(jazz[count:])
train+=(metal[count:])
train+=(hiphop[count:])
train+=(blues[count:])
train+=(classical[count:])
train+=(country[count:])
train+=(pop[count:])
train+=(disco[count:])
train+=(rock[count:])

print 'train : ', len(train)
print 'valid : ', len(valid)

valid = pd.Series(valid)
train = pd.Series(train)



i = 0
def cp(source, target):
    shutil.copy2(source, target)
    global i
    i+=1
    print i
train.apply(lambda x : cp(path+x, train_path+x))
valid.apply(lambda x : cp(path+x, valid_path+x))
print 'END'




