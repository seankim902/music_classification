# -*- coding: utf-8 -*-
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
from keras import activations, initializations

import optimizer
import cPickle

import numpy as np


def save_tparams(tparams, path):
    with open(path,'wb') as f:
        for params in tparams:
            cPickle.dump(params.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_tparams(tparams, path):
    with open(path,'rb') as f:
          for i in range(len(tparams)):
              tparams[i].set_value(cPickle.load(f))
    return tparams

def get_minibatch_indices(n, batch_size, shuffle=False):

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return minibatches


def concatenate(tensor_list, axis=0):

    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
    
    
def dropout_layer(state_before, use_noise, trng, p):
    ratio = 1. - p
    proj = T.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=ratio, n=1, dtype=state_before.dtype), # for training..
            state_before * ratio)
    return proj
    
    
class RNN_GRU:
    def __init__(self, y_vocab, dim_word, dim, dim_ctx):
    
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512
        self.dim_ctx = dim_ctx  # 512
        
        ### image Embedding
        self.W_img_emb = initializations.glorot_uniform((self.dim_ctx, self.dim))     
        self.b_img_emb = initializations.zero((self.dim))


        
        ### enc forward GRU ###
        self.W_gru = initializations.glorot_uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.glorot_uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.W_gru_cdd = initializations.glorot_uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.glorot_uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim))       
        ### prediction ###
        self.W_pred = initializations.glorot_uniform((self.dim, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_img_emb, self.b_img_emb,
                       self.W_gru, self.U_gru, self.b_gru,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd,
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, init_state):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru 
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd 
        
        def _step(x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim) )
            u = T.nnet.sigmoid(_slice(preact, 1, dim) )
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_  # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
    
            return h#, r, u, preact, preactx
        seqs = [state_below_, state_belowx]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state], #T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.tensor3('x', dtype = 'float32')
        y = T.matrix('y', dtype = 'int32')
        img = T.matrix('img', dtype = 'float32')
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        init_state = T.dot(img, self.W_img_emb) + self.b_img_emb
        emb = x

        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, init_state)
        

        # hidden 들의 평균
        proj = proj.mean(axis=0)
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        

        if dropout is not None :
            proj = dropout_layer(proj, use_noise, trng, dropout)
            
            
        output = T.dot(proj, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
        
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)

        '''
        decay_c = 0.000001
        # add L2 regularization costs
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for vv in self.params:
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay
    
        '''
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, img, y, cost, updates, prediction
        
        
 
        
    def train(self, train_x, train_img, train_y,
              valid_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout = None,
              batch_size=16,
              epoch=100,
              save=None):

        best_val = 0.
        n_train = train_x.shape[0]
        
        trng, use_noise, x, img, y, cost, updates, prediction = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        train_model = theano.function(inputs=[x, img, y],
                                      outputs=cost,
                                      updates=updates)
                                       
        if valid is not None:
            valid_model = theano.function(inputs=[x, img],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[0], batch_size)
            
        for i in xrange(epoch):
            
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
           
            for j, indices in enumerate(batch_indices):
                
                x = [ train_x[t,:,:] for t in indices]
                x = np.array(x).swapaxes(0,1)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t,:] for t in indices]
                img = np.array(img)
                
                minibatch_avg_cost = train_model(x, img, y)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', j+1, '\' in epoch \'', (i+1) ,'\' ]'
            
            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[t,:,:] for t in valid_indices]
                        val_x = np.array(val_x).swapaxes(0,1)
                        val_batch_imgs = [ valid_img[t,:] for t in valid_indices]
                        val_batch_imgs = np.array(val_batch_imgs)
                        
                        valid_prediction += valid_model(val_x, val_batch_imgs).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best_val :
                        best_val = (float(correct) / len(valid_prediction))
            '''        
            if save is not None:
                if (i+1) % save == 0:
                    print 'save param..',
                    save_tparams(self.params, 'model.pkl')
                    print 'Done'
            '''        
                    
        print 'best : ', best_val
                    
                    
                    
    
    
class BIRNN_GRU:
    def __init__(self, y_vocab, dim_word, dim, dim_ctx):
    
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512
        self.dim_ctx = dim_ctx  # 512
        
        ### image Embedding
        self.W_img_emb = initializations.glorot_uniform((self.dim_ctx, self.dim))     
        self.b_img_emb = initializations.zero((self.dim))


        
        ### enc forward GRU ###
        self.W_gru = initializations.glorot_uniform((self.dim_word, self.dim * 2))
        self.U_gru = initializations.glorot_uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.W_gru_cdd = initializations.glorot_uniform((self.dim_word, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.glorot_uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim))       
        ### prediction ###
        self.W_pred = initializations.glorot_uniform((self.dim * 2, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_img_emb, self.b_img_emb,
                       self.W_gru, self.U_gru, self.b_gru,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd,
                       self.W_pred, self.b_pred]

    def gru_layer(self, state_below, init_state):
        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru 
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd 
        
        def _step(x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim) )
            u = T.nnet.sigmoid(_slice(preact, 1, dim) )
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_  # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
    
            return h#, r, u, preact, preactx
        seqs = [state_below_, state_belowx]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state], #T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.tensor3('x', dtype = 'float32')
        y = T.matrix('y', dtype = 'int32')
        img = T.matrix('img', dtype = 'float32')
        


        init_state = T.dot(img, self.W_img_emb) + self.b_img_emb
        emb = x
        embr = x.swapaxes(0,1)[::-1].swapaxes(0,1)
        

        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, init_state)
        projr = self.gru_layer(embr, init_state)
        
        proj = concatenate([proj, projr[::-1]], axis=proj.ndim-1)

        # hidden 들의 평균
        proj = proj.mean(axis=0)
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        

        if dropout is not None :
            proj = dropout_layer(proj, use_noise, trng, dropout)
            
            
        output = T.dot(proj, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
        
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)

        '''
        decay_c = 0.000001
        # add L2 regularization costs
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for vv in self.params:
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay
    
        '''
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, img, y, cost, updates, prediction, probs
        
        
 
        
    def train(self, train_x, train_img, train_y,
              valid_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout = None,
              batch_size=16,
              epoch=100,
              save=None):

        best_val = 0.
        n_train = train_x.shape[0]
        
        trng, use_noise, x, img, y, cost, updates, prediction, _ = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        train_model = theano.function(inputs=[x, img, y],
                                      outputs=cost,
                                      updates=updates)
                                       
        if valid is not None:
            valid_model = theano.function(inputs=[x, img],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[0], batch_size)
            
        for i in xrange(epoch):
            
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
           
            for j, indices in enumerate(batch_indices):
                
                x = [ train_x[t,:,:] for t in indices]
                x = np.array(x).swapaxes(0,1)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t,:] for t in indices]
                img = np.array(img)
                
                minibatch_avg_cost = train_model(x, img, y)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', j+1, '\' in epoch \'', (i+1) ,'\' ]'
            
            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[t,:,:] for t in valid_indices]
                        val_x = np.array(val_x).swapaxes(0,1)
                        val_batch_imgs = [ valid_img[t,:] for t in valid_indices]
                        val_batch_imgs = np.array(val_batch_imgs)
                        
                        valid_prediction += valid_model(val_x, val_batch_imgs).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best_val :
                        best_val = (float(correct) / len(valid_prediction))
                    

                        print 'save param..',
                        save_tparams(self.params, 'model.pkl')
                        print 'Done'
                    
                    
        print 'best : ', best_val
                    

                    
    def prediction(self, test_x, test_img, test_y,
              lr=0.001,
              batch_size=16):

        load_tparams(self.params, 'model.pkl')
        n_test = test_x.shape[0]
        
        trng, use_noise, x, img, y, _, _, prediction, probs = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        test_model = theano.function(inputs=[x, img],
                                      outputs=[prediction, probs])
                                       

        batch_indices=get_minibatch_indices(n_test, batch_size, shuffle=False)

        prediction = []
        probs = []
        
        for j, indices in enumerate(batch_indices):
            
            x = [ test_x[t,:,:] for t in indices]
            x = np.array(x).swapaxes(0,1)
            img = [ test_img[t,:] for t in indices]
            img = np.array(img)
            
            result = test_model(x, img)
            prediction += result[0].tolist()
            probs += result[1].tolist()

        return prediction, probs
                    
class BIRNN_LSTM:
    def __init__(self, y_vocab, dim_word, dim, dim_ctx):

        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512
        self.dim_ctx = dim_ctx  # 512
        
        ### 
        ### initial context - image Embedding
        self.W_hidden_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_hidden_init = initializations.zero((self.dim))
        self.W_memory_init = initializations.uniform((self.dim_ctx, self.dim))     
        self.b_memory_init = initializations.zero((self.dim))


        
        ### enc forward GRU ###

        self.W_lstm = initializations.uniform((self.dim_word, self.dim * 4))
        self.U_lstm = initializations.uniform((self.dim, self.dim * 4))
        self.b_lstm = initializations.zero((self.dim * 4))
        
        
        ### prediction ###
        self.W_pred = initializations.uniform((self.dim * 2, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_hidden_init, self.b_hidden_init,self.W_memory_init, self.b_memory_init,
                       self.W_lstm, self.U_lstm, self.b_lstm,
                       self.W_pred, self.b_pred]

    def lstm_layer(self, state_below,  init_state=None, init_memory=None):


        #state_below : step * sample * dim
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
 
        if init_state is None :
            init_state = T.alloc(0., n_samples, dim)
        if init_memory is None:
            init_memory = T.alloc(0., n_samples, dim)   
            
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_lstm) + self.b_lstm
        
        
        def _step(x_, h_, c_, U):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            i = _slice(preact, 0, dim)
            f = _slice(preact, 1, dim)
            o = _slice(preact, 2, dim)
    
            i = T.nnet.sigmoid(i)
            f = T.nnet.sigmoid(f)
            o = T.nnet.sigmoid(o)
            c = T.tanh(_slice(preact, 3, dim))
    
            c = f * c_ + i * c
            h = o * T.tanh(c)
    

    
            return [h, c]#
        seqs = [state_below_]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state, init_memory], #T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_lstm],
                                    name='lstm_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.tensor3('x', dtype = 'float32')
        y = T.matrix('y', dtype = 'int32')
        img = T.matrix('img', dtype = 'float32')
        
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        
        init_state = T.dot(img, self.W_hidden_init) + self.b_hidden_init
        init_memory = T.dot(img, self.W_memory_init) + self.b_memory_init
        
        
        emb = x
        embr = x.swapaxes(0,1)[::-1].swapaxes(0,1)



        # proj : gru hidden 들의 리스트   
        proj = self.lstm_layer(emb, init_state=init_state, init_memory=init_memory)[0]
        projr = self.lstm_layer(embr, init_state=init_state, init_memory=init_memory)[0]
        
        proj = concatenate([proj, projr[::-1]], axis=proj.ndim-1)

        # hidden 들의 평균
        proj = proj.mean(axis=0)
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        

        if dropout is not None :
            proj = dropout_layer(proj, use_noise, trng, dropout)
            
            
        output = T.dot(proj, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-8
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
        
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)

        '''
        decay_c = 0.000001
        # add L2 regularization costs
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for vv in self.params:
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay
    
        '''
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, img, y, cost, updates, prediction
        
        
 
        
    def train(self, train_x, train_img, train_y,
              valid_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout = None,
              batch_size=16,
              epoch=100,
              save=None):

        best_val = 0.
        n_train = train_x.shape[0]
        
        trng, use_noise, x, img, y, cost, updates, prediction = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        train_model = theano.function(inputs=[x, img, y],
                                      outputs=cost,
                                      updates=updates)
                                       
        if valid is not None:
            valid_model = theano.function(inputs=[x, img],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[0], batch_size)
            
        for i in xrange(epoch):
            
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
           
            for j, indices in enumerate(batch_indices):
                
                x = [ train_x[t,:,:] for t in indices]
                x = np.array(x).swapaxes(0,1)
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t,:] for t in indices]
                img = np.array(img)
                
                minibatch_avg_cost = train_model(x, img, y)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', j+1, '\' in epoch \'', (i+1) ,'\' ]'
            
            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[t,:,:] for t in valid_indices]
                        val_x = np.array(val_x).swapaxes(0,1)
                        val_batch_imgs = [ valid_img[t,:] for t in valid_indices]
                        val_batch_imgs = np.array(val_batch_imgs)
                        
                        valid_prediction += valid_model(val_x, val_batch_imgs).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best_val :
                        best_val = (float(correct) / len(valid_prediction))
                    
                        print 'save param..',
                        save_tparams(self.params, 'model.pkl')
                        print 'Done'
                    
                    
        print 'best : ', best_val
                    
                    


    
    
class BIRNN_GRU2:
    def __init__(self, y_vocab, dim_word, dim, dim_ctx):
    
        self.y_vocab = y_vocab  # 430
        self.dim_word = dim_word # 1024
        self.dim = dim  # 512
        self.dim_ctx = dim_ctx  # 512
        self.emb_dim = 512
        
        
        ### image Embedding
        self.W_img_emb = initializations.glorot_uniform((self.dim_ctx, self.emb_dim))     
        self.b_img_emb = initializations.zero((self.emb_dim))   

        self.W_fr_emb = initializations.glorot_uniform((self.dim_word, self.emb_dim))     
        self.b_fr_emb = initializations.zero((self.emb_dim))  
        
        ### enc forward GRU ###
        self.W_gru = initializations.glorot_uniform((self.emb_dim, self.dim * 2))
        self.U_gru = initializations.glorot_uniform((self.dim, self.dim * 2))
        self.b_gru = initializations.zero((self.dim * 2))
        self.W_gru_cdd = initializations.glorot_uniform((self.emb_dim, self.dim)) # cdd : candidate
        self.U_gru_cdd = initializations.glorot_uniform((self.dim, self.dim))
        self.b_gru_cdd = initializations.zero((self.dim))       
        ### prediction ###
        self.W_pred = initializations.glorot_uniform((self.dim * 2, self.y_vocab))
        self.b_pred = initializations.zero((self.y_vocab))


        self.params = [self.W_img_emb, self.W_fr_emb, self.b_img_emb, self.b_fr_emb,
                       self.W_gru, self.U_gru, self.b_gru,
                       self.W_gru_cdd, self.U_gru_cdd, self.b_gru_cdd,
                       self.W_pred, self.b_pred]

    def gru_layer(self, emb, img):
        #state_below : (1+step) * sample * dim

#        x = T.tensor3('x', dtype = 'float32') # step * sample * dim
#        y = T.matrix('y', dtype = 'int32')
#        img = T.matrix('img', dtype = 'float32') #  sample * dim
        
        emb = T.dot(emb, self.W_fr_emb) + self.b_fr_emb 
        img = T.dot(img, self.W_img_emb) + self.b_img_emb 
        state_below = concatenate([img, emb])

        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru 
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd 
        
        def _step(x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim) )
            u = T.nnet.sigmoid(_slice(preact, 1, dim) )
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_  # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
    
            return h#, r, u, preact, preactx
        seqs = [state_below_, state_belowx]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
    def gru_cond_layer(self, embr, img):
        #state_below : step * sample * dim


        embr = T.dot(embr, self.W_fr_emb) + self.b_fr_emb 
        img = T.dot(img, self.W_img_emb) + self.b_img_emb 
        state_below = concatenate([embr, img])
        
        
        nsteps = state_below.shape[0]
        n_samples = state_below.shape[1]
        
        dim = self.dim

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                print '_x.ndim : ' , _x.ndim
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]
    
        # step * samples * dim
        state_below_ = T.dot(state_below, self.W_gru) + self.b_gru 
        state_belowx = T.dot(state_below, self.W_gru_cdd) + self.b_gru_cdd 
        
        def _step(x_, xx_, h_, U, Ux):
            '''
            m_ : (samples,)
            x_, h_ : samples * dimensions   
            '''
            preact = T.dot(h_, U)
            preact += x_ # samples * 1024
    
            r = T.nnet.sigmoid(_slice(preact, 0, dim) )
            u = T.nnet.sigmoid(_slice(preact, 1, dim) )
    
            preactx = T.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_  # samples * 512
    
            h = T.tanh(preactx)
    
            h = u * h_ + (1. - u) * h
    
            return h#, r, u, preact, preactx
        seqs = [state_below_, state_belowx]
    
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [T.alloc(0., n_samples, dim)],
                                    non_sequences = [self.U_gru, self.U_gru_cdd],
                                    name='gru_layer',
                                    n_steps=nsteps)
        return rval
        
                            
    def build_model(self, lr=0.001, dropout=None):
    
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
    
        # description string: #words x #samples


        x = T.tensor3('x', dtype = 'float32') # step * sample * 5555
        y = T.matrix('y', dtype = 'int32')
        img = T.tensor3('img', dtype = 'float32') #  1*sample * 4096
        



#        T.set_subtensor(img_t3[0], img)

 #       emb=theano.tensor.concatenate([img_t3,x])
        emb = x
        embr = x[::-1]

        # proj : gru hidden 들의 리스트   
        proj = self.gru_layer(emb, img)
        projr = self.gru_cond_layer(embr, img)
        
        proj = concatenate([proj, projr[::-1]], axis=proj.ndim-1)

        # hidden 들의 평균
        proj = proj.mean(axis=0)
        
        # 마지막 hidden
        #proj = proj[-1]  # sample * dim
        

        if dropout is not None :
            proj = dropout_layer(proj, use_noise, trng, dropout)
            
            
        output = T.dot(proj, self.W_pred) + self.b_pred
        
        probs = T.nnet.softmax(output)
        prediction = probs.argmax(axis=1)
        
        ## avoid NaN
        epsilon = 1.0e-9
        probs = T.clip(probs, epsilon, 1.0 - epsilon)
        probs /= probs.sum(axis=-1, keepdims=True)
        ## avoid NaN
        
        cost = T.nnet.categorical_crossentropy(probs, y)
        cost = T.mean(cost)

        '''
        decay_c = 0.000001
        # add L2 regularization costs
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for vv in self.params:
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay
    
        '''
        
        updates = optimizer.adam(cost=cost, params=self.params, lr=lr)

        return trng, use_noise, x, img, y, cost, updates, prediction, probs
        
        
 
        
    def train(self, train_x, train_img, train_y,
              valid_x=None, valid_img=None, valid_y=None,
              valid=None,
              lr=0.001,
              dropout = None,
              batch_size=16,
              epoch=100,
              save=None):
        
        train_img = train_img[:,None,:]
        valid_img = valid_img[:,None,:]
        
        best_val = 0.
        n_train = train_x.shape[0]
        
        trng, use_noise, x, img, y, cost, updates, prediction, _ = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb

        train_model = theano.function(inputs=[x, img, y],
                                      outputs=cost,
                                      updates=updates)
        if valid is not None:
            valid_model = theano.function(inputs=[x, img],
                                          outputs=prediction)   
            valid_batch_indices = get_minibatch_indices(valid_x.shape[0], batch_size)
        for i in xrange(epoch):
            
            batch_indices=get_minibatch_indices(n_train, batch_size, shuffle=True)
           
            for j, indices in enumerate(batch_indices):
                
                x = [ train_x[t,:,:] for t in indices]
                x = np.array(x).swapaxes(0,1)  # step * sample * 5512
                y = [ train_y[t,:] for t in indices]
                y = np.array(y)
                img = [ train_img[t,:,:] for t in indices]
                img = np.array(img).swapaxes(0,1)  # 1 * sample * 4096

                minibatch_avg_cost = train_model(x, img, y)
                print 'cost : ' , minibatch_avg_cost, ' [ mini batch \'', j+1, '\' in epoch \'', (i+1) ,'\' ]'
            
            # validation  
            if valid is not None:
                if (i+1) % valid == 0:
                    use_noise.set_value(0.)

                    valid_prediction = []
                    for k, valid_indices in enumerate(valid_batch_indices):
                        
                        val_x = [ valid_x[t,:,:] for t in valid_indices]
                        val_x = np.array(val_x).swapaxes(0,1)
                        val_batch_imgs = [ valid_img[t,:,:] for t in valid_indices]
                        val_batch_imgs = np.array(val_batch_imgs).swapaxes(0,1)
                        
                        valid_prediction += valid_model(val_x, val_batch_imgs).tolist()
                    correct = 0 
                    for l in range(len(valid_prediction)):
                        if valid_prediction[l]==valid_y[l] : 
                            correct += 1
                    print '## valid accuracy : ', float(correct) / len(valid_prediction)
                    if (float(correct) / len(valid_prediction)) > best_val :
                        best_val = (float(correct) / len(valid_prediction))
                    
                        '''
                        print 'save param..',
                        save_tparams(self.params, 'model.pkl')
                        print 'Done'
                        '''
                    
        print 'best : ', best_val
                    

                    
    def prediction(self, test_x, test_img, test_y,
              lr=0.001,
              batch_size=16):

        load_tparams(self.params, 'model.pkl')
        n_test = test_x.shape[0]
        
        trng, use_noise, x, img, y, _, _, prediction, probs = self.build_model(lr)
        # x : step * sample * dim
        # x_mask : step * sample
        # y : sample * emb


        test_model = theano.function(inputs=[x, img],
                                      outputs=[prediction, probs])
                                       

        batch_indices=get_minibatch_indices(n_test, batch_size, shuffle=False)

        prediction = []
        probs = []
        
        for j, indices in enumerate(batch_indices):
            
            x = [ test_x[t,:,:] for t in indices]
            x = np.array(x).swapaxes(0,1)
            img = [ test_img[t,:] for t in indices]
            img = np.array(img)
            
            result = test_model(x, img)
            prediction += result[0].tolist()
            probs += result[1].tolist()

        return prediction, probs
   
                   
      