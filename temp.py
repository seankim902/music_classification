# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.io.wavfile as wav

import pandas as pd




def convert_np_audio_to_sample_blocks(song_np, block_size):
	block_lists = []
	total_samples = song_np.shape[0]
	num_samples_so_far = 0
	while(num_samples_so_far < total_samples):
		block = song_np[num_samples_so_far:num_samples_so_far+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0],))
			block = np.concatenate((block, padding))
		block_lists.append(block)
		num_samples_so_far += block_size
	return block_lists
 


def time_blocks_to_fft_blocks(blocks_time_domain):
	fft_blocks = []
	for block in blocks_time_domain:
		fft_block = np.fft.fft(block)
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks
 
def read_wav_as_np(filename):
	data = wav.read(filename)
	np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit input to [-1, 1] range
	#np_arr = np.array(np_arr)
	return np_arr, data[0]

def load_training_example(filename, block_size=2048, useTimeDomain=False):
	data, bitrate = read_wav_as_np(filename)
	x_t = convert_np_audio_to_sample_blocks(data, block_size)
	y_t = filename.split('/')[-1].split('.')[0]

 
	if useTimeDomain:
		return x_t, y_t
	X = time_blocks_to_fft_blocks(x_t)
	Y = y_t
	return X, Y
 
 
def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, useTimeDomain=True):
	files = []
	for file in os.listdir(directory):
		if file.endswith('.wav'):
			files.append(directory+file)
	chunks_X = []
	y_data = []
	num_files = len(files)

	for file_idx in xrange(num_files):
		file = files[file_idx]
		print file_idx
		X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
		y_data.append(Y)
		cur_seq = 0
		chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
  		if len(X[cur_seq:cur_seq+max_seq_len]) != 120:
			print file, len(X[cur_seq:cur_seq+max_seq_len])
	num_examples = len(chunks_X)
	num_dims_out = block_size * 2
	if(useTimeDomain):
		num_dims_out = block_size
	out_shape = (num_examples, max_seq_len, num_dims_out)
	x_data = np.zeros(out_shape)
 
	for n in xrange(num_examples):
		for i in xrange(max_seq_len):
			x_data[n][i] = chunks_X[n][i]
		print 'Saved example ', (n+1), ' / ',num_examples
	print 'Flushing to disk...'
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance
	print 'shape : ', x_data.shape

	#np.save(out_file+'_mean', mean_x)
	#np.save(out_file+'_var', std_x)
	#np.save(out_file+'_x', x_data)
     
	return x_data, y_data, files


freq = 22050
clip_len = 28 		#length of clips for training. Defined in seconds
block_size = freq / 2 #block sizes used for training - this defines the size of our input state
max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
print max_seq_len


x_data, y_data, files = convert_wav_files_to_nptensor('/home/seonhoon/Desktop/workspace/music/data/valid/', block_size, max_seq_len, 'temp.npy')
new_x_data=[]
for i in range(len(x_data)):
    print x_data[i].shape
    new_x_data.append(x_data[i].flatten())


train=pd.DataFrame()

genre={'reggae':0,'jazz':1,'metal':2,'hiphop':3,'blues':4,'classical':5,'country':6,'pop':7,'disco':8,'rock':9}

train['x']=new_x_data
train['genre']=y_data
train['file']=files
y_data2=[]
for i in range(len(y_data)):
    y_data2.append( genre[y_data[i]])
train['y']=y_data2

train.to_pickle('/home/seonhoon/Desktop/workspace/music/data/valid.pkl')
print 'end'