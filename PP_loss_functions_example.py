"""
@author: Carl Southall     carlsouthall.com    carl.southall@bcu.ac.uk     https://github.com/CarlSouthall
"""

# This file contains an example implementation of the multiple time-step loss functions presented in [1].
# Included are:
#             1. MI: multiple individual 
#             2. MMD: multiple difference using mean squared
#             3. WMD: multiple difference using weighted cross entropy

# The cnnSA3F5 network used is also used in [2].

#  [1] Southall, Carl, Ryan Stables and Jason Hockman. 2018. 
#  Improving Peak-picking Using Multiple Time-step Loss Functions. 
#  In Proceedings of the 19th International Society for Music Information
#  Retrieval Conference (ISMIR), Paris, France, 2018.
#
#  [2] Southall, Carl, Ryan Stables and Jason Hockman. 2018. 
#  Player Vs Transcriber: A Game Approach to Data Manipulation for Automatic
#  Drum Transcription. In Proceedings of the 19th International Society for
#  Music Information Retrieval Conference (ISMIR), Paris, France, 2018.
#
#
#################################################################################
# packages and variables

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn   
import os

eps=np.finfo(float).eps
alpha = 0.5
gamma = 2
#################################################################################

## loss functions

def cross_entropy_form(pred,lab):
    out=(lab * tf.log(pred))+((1-lab)*(tf.log(1-pred)))

    return out
def cross_entropy_focal_form(pred,lab):
    out =  ( lab *alpha * tf.pow(1 -  pred, gamma) * tf.log( pred) + (1-alpha) * (1-lab) * tf.pow(pred, gamma) * tf.log(1 - pred))
    return out
#-1 * labels_ *self._alpha * mx.nd.power(1 -  pro + self.eps, self._gamma) * mx.nd.log( pro+self.eps) - (1-self._alpha) * (1-labels_) * mx.nd.power(pro + self.eps, self._gamma) * mx.nd.log(1 - pro + self.eps)


def mean_squared_form(pred,lab):
    out=tf.square(lab-pred)
    return out

def mean_squared_peak_dif_form(pred1,lab1,pred2,lab2):
    out=tf.square((lab1-lab2)-(pred1-pred2))
    return out

def weighted_cross_entropy_peak_dif_form(pred1,lab1,pred2,lab2,FP_weighting,eps=np.finfo(float).eps):
    FN_weighting=1-FP_weighting
    out=FN_weighting*(tf.abs(lab1-lab2)*(tf.log(tf.abs(pred1-pred2)+eps)))+FP_weighting*((1-tf.abs(lab1-lab2))*(tf.log(1-tf.abs(pred1-pred2)+eps)))
    return out 

def MI(preds,labs,weighting,seq_len):
   stand_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,1:seq_len+1],labs[:,1:seq_len+1]),dtype=(tf.float32))    
   previous_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,:seq_len],labs[:,:seq_len]),dtype=(tf.float32))
   future_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]), (preds[:,2:seq_len+2],labs[:,2:seq_len+2]),dtype=(tf.float32))
   cost=tf.reduce_mean(tf.add(tf.reshape(stand_ce,[-1]),(weighting*tf.add(tf.reshape(future_ce,[-1]),tf.reshape(previous_ce,[-1])))))

   return cost

def MMD(preds,labs,weighting,seq_len):      
   stand_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,1:seq_len+1],labs[:,1:seq_len+1]),dtype=(tf.float32))    
   previous_ce=tf.map_fn(lambda x: -tf.reduce_sum(mean_squared_peak_dif_form(x[0],x[1],x[2],x[3]), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,:seq_len],labs[:,:seq_len]),dtype=(tf.float32))
   future_ce=tf.map_fn(lambda x: -tf.reduce_sum(mean_squared_peak_dif_form(x[0],x[1],x[2],x[3]), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,2:seq_len+2],labs[:,2:seq_len+2]),dtype=(tf.float32))
   cost=tf.reduce_mean(tf.add(tf.reshape(stand_ce,[-1]),(weighting*tf.add(tf.reshape(future_ce,[-1]),tf.reshape(previous_ce,[-1])))))
   return cost

def WMD(preds,labs,weighting,seq_len,FP_weighting_,cross_entropy_fun):      
   stand_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_fun(x[0],x[1]), reduction_indices=[1]),(preds[:,1:seq_len+1],labs[:,1:seq_len+1]),dtype=(tf.float32))    
   previous_ce=tf.map_fn(lambda x: -tf.reduce_sum(weighted_cross_entropy_peak_dif_form(x[0],x[1],x[2],x[3],FP_weighting_), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,:seq_len],labs[:,:seq_len]),dtype=(tf.float32))
   future_ce=tf.map_fn(lambda x: -tf.reduce_sum(weighted_cross_entropy_peak_dif_form(x[0],x[1],x[2],x[3],FP_weighting_), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,2:seq_len+2],labs[:,2:seq_len+2]),dtype=(tf.float32))
   cost=tf.reduce_mean(tf.add(tf.reshape(stand_ce,[-1]),(weighting*tf.add(tf.reshape(future_ce,[-1]),tf.reshape(previous_ce,[-1])))))
   return cost



#################################################################################

## cnnSA3B-F5 model used in [1,2]
            
class cnnSA3BF5():
#     
     def __init__(self,training_data=[], training_labels=[], validation_data=[], validation_labels=[], network_save_filename=[], minimum_epoch=10, maximum_epoch=20, n_hidden=[20,20], n_classes=3, attention_number=3, dropout=0.75,  learning_rate=0.003, snippet_length=100, cost_type='CE',batch_size=1000,input_feature_size=84*11,conv_filter_shapes=[[3,3,1,32],[3,3,32,64]], conv_strides=[[1,1,1,1],[1,1,1,1]], pool_window_sizes=[[1,1,2,1],[1,1,2,1]],pad='SAME',save_location=[]):         
         self.features=training_data
         self.targ=training_labels
         self.val=validation_data
         self.val_targ=validation_labels
         self.n_hidden=n_hidden
         self.n_layers=len(self.n_hidden)
         self.filename=network_save_filename
         self.dropout=dropout
         self.learning_rate=learning_rate
         self.n_classes=n_classes
         self.minimum_epoch=minimum_epoch
         self.maximum_epoch=maximum_epoch
         self.num_batch=int(len(self.features)/batch_size)
         self.val_num_batch=int(len(self.val)/batch_size)
         self.batch_size=batch_size
         self.attention_number=attention_number
         self.cost_type=cost_type
         self.input_feature_size=input_feature_size
         self.batch=np.zeros((self.batch_size,self.input_feature_size))
         self.batch_targ=np.zeros((self.batch_size,self.n_classes))
         self.snippet_length=snippet_length
         self.num_seqs=int(self.batch_size/self.snippet_length)
         self.conv_filter_shapes=conv_filter_shapes
         self.conv_strides=conv_strides
         self.pool_window_sizes=pool_window_sizes
         self.pool_strides=self.pool_window_sizes
         self.conv_layer_out=[]
         self.fc_layer_out=[]
         self.w_fc=[]
         self.b_fc=[]
         self.h_fc=[]
         self.w_conv=[]
         self.b_conv=[]
         self.w_conv_gated=[]
         self.b_conv_gated=[]
         self.h_conv=[]
         self.h_pool=[]
         self.h_drop_batch=[]
         self.pad=pad
         self.save_location=save_location

     def cell_create(self,scope_name):
         with tf.variable_scope(scope_name):
             if int(scope_name)==1:
                 cells = rnn.DropoutWrapper(rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[int(scope_name)-1]) for i in range(1)], state_is_tuple=True), input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph)
             else:
                 cells = rnn.DropoutWrapper(rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[int(scope_name)-1+i]) for i in range(self.n_layers-1)], state_is_tuple=True),input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph)      
         return cells
     
     def weight_bias_init(self):
               
            self.biases = tf.Variable(tf.zeros([self.n_classes]))                     
            self.weights =tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2, self.n_classes]))

            
     def cell_create_norm(self):
         cells = rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
         cells = rnn.DropoutWrapper(cells, input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph) 
         return cells
           
     def attention_weight_init(self,num):
         if num==0:
             self.attention_weights=[tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*4,self.n_hidden[(len(self.n_hidden)-1)]*2]))]
             self.sm_attention_weights=[tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2,self.n_hidden[(len(self.n_hidden)-1)]*2]))]
         if num>0:
             self.attention_weights.append(tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*4,self.n_hidden[(len(self.n_hidden)-1)]*2])))
             self.sm_attention_weights.append(tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2,self.n_hidden[(len(self.n_hidden)-1)]*2])))
                     
             
     def conv2d(self,data, weights, conv_strides, pad):
         return tf.nn.conv2d(data, weights, strides=conv_strides, padding=pad)
     
     def max_pool(self,data, max_pool_window, max_strides, pad):
        return tf.nn.max_pool(data, ksize=max_pool_window,
                            strides=max_strides, padding=pad)
        
     def weight_init(self,weight_shape):
        weight=tf.Variable(tf.truncated_normal(weight_shape))    
        return weight
        
     def bias_init(self,bias_shape,):   
        bias=tf.Variable(tf.constant(0.1, shape=bias_shape))
        return bias
    
     def batch_dropout(self,data):
        batch_mean, batch_var=tf.nn.moments(data,[0])
        scale=tf.Variable(tf.ones([self.batch_size,data.get_shape()[1],data.get_shape()[2],data.get_shape()[3]]))
        beta=tf.Variable(tf.zeros([self.batch_size,data.get_shape()[1],data.get_shape()[2],data.get_shape()[3]])) ### removed for quicker implementation
        h_poolb=tf.nn.batch_normalization(data,batch_mean,batch_var,beta,scale,1e-3)
        return tf.nn.dropout(data, self.dropout_ph)

     def gated_linear_layer(self,inputs, gates, name=None):

        activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

        return activation
        
     def conv_2dlayer(self,layer_num):
        self.w_conv.append(self.weight_init(self.conv_filter_shapes[layer_num]))
        self.b_conv.append(self.bias_init([self.conv_filter_shapes[layer_num][3]]))
        self.h1 = self.conv2d(self.conv_layer_out[layer_num], self.w_conv[layer_num], self.conv_strides[layer_num], self.pad) + self.b_conv[layer_num]
        self.w_conv_gated.append(self.weight_init(self.conv_filter_shapes[layer_num]))
        self.b_conv_gated.append(self.bias_init([self.conv_filter_shapes[layer_num][3]]))
        self.h1_gated = self.conv2d(self.conv_layer_out[layer_num], self.w_conv_gated[layer_num], self.conv_strides[layer_num], self.pad) + self.b_conv_gated[layer_num]
#       self.h1_glu = self.gated_linear_layer(inputs=self.h1, gates=self.h1_gated)
#       self.h_conv.append(tf.nn.relu(self.h1_glu))
        self.h_conv.append(self.gated_linear_layer(inputs=self.h1, gates=self.h1_gated))
        self.h_pool.append(self.max_pool(self.h_conv[layer_num],self.pool_window_sizes[layer_num],self.pool_strides[layer_num],self.pad))       
        self.conv_layer_out.append(self.batch_dropout(self.h_pool[layer_num]))  
        self.conv_layer_out.append(self.h_pool[layer_num])  

     def reshape_layer(self):
            convout=self.conv_layer_out[len(self.conv_layer_out)-1]
            self.fc_layer_out=tf.reshape(convout, [self.num_seqs,self.seq_len,self.conv_layer_out[len(self.conv_layer_out)-1].get_shape()[1]*self.conv_layer_out[len(self.conv_layer_out)-1].get_shape()[2]*self.conv_layer_out[len(self.conv_layer_out)-1].get_shape()[3]])
            
     def create(self):
       
       tf.reset_default_graph()
       self.x_ph = tf.placeholder("float32", [None, None, self.batch.shape[1]])
       self.y_ph = tf.placeholder("float32", [None, None, self.batch_targ.shape[1]]) 
       self.seq=tf.placeholder("int32")
       self.num_seqs=tf.placeholder("int32")
       self.seq_len=tf.placeholder("int32")
       self.dropout_ph = tf.placeholder("float32")
       
       self.conv_layer_out.append(tf.expand_dims(tf.reshape(self.x_ph,[-1,11,int(self.input_feature_size/11)]),3))
       for i in range(len(self.conv_filter_shapes)):
            self.conv_2dlayer(i)
       self.reshape_layer()
       self.fw_cell=self.cell_create('1')
       self.fw_cell2=self.cell_create('2')
       self.weight_bias_init()
       
       self.bw_cell=self.cell_create('1')
       self.bw_cell2=self.cell_create('2') 
       with tf.variable_scope('1'):

           self.outputs, self.states= tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.fc_layer_out,
                                             sequence_length=self.seq,dtype=tf.float32)
                                          
       self.first_out=tf.concat((self.outputs[0],self.outputs[1]),2)
       with tf.variable_scope('2'):
           self.outputs2, self.states2= tf.nn.bidirectional_dynamic_rnn(self.fw_cell2, self.bw_cell2, self.first_out,
                                             sequence_length=self.seq,dtype=tf.float32)
       self.second_out=tf.concat((self.outputs2[0],self.outputs2[1]),2)

       for i in range((self.attention_number*2)+1):
           self.attention_weight_init(i)
        
       self.zero_pad_second_out=tf.map_fn(lambda x:tf.pad(tf.squeeze(x),[[self.attention_number,self.attention_number],[0,0]]),self.second_out)
       self.first_out_reshape=tf.reshape(self.first_out,[-1,self.n_hidden[self.n_layers-1]*2])
       self.zero_pad_second_out_reshape=[]
       self.attention_m=[]
       for j in range((self.attention_number*2)+1):
           self.zero_pad_second_out_reshape.append(tf.reshape(tf.slice(self.zero_pad_second_out,[0,j,0],[self.num_seqs,self.seq_len,self.n_hidden[self.n_layers-1]*2]),[-1,self.n_hidden[self.n_layers-1]*2]))
           self.attention_m.append(tf.tanh(tf.matmul(tf.concat((self.zero_pad_second_out_reshape[j],self.first_out_reshape),1),self.attention_weights[j])))
       self.attention_s=tf.nn.softmax(tf.stack([tf.matmul(self.attention_m[j],self.sm_attention_weights[j]) for j in range(self.attention_number*2+1)]),0)
       self.attention_z=tf.reduce_sum([self.attention_s[j]*self.zero_pad_second_out_reshape[j] for j in range(self.attention_number*2+1)],0)
       self.attention_z_reshape=tf.reshape(self.attention_z,[self.num_seqs,self.seq_len,self.n_hidden[self.n_layers-1]*2])
       self.presoft=tf.map_fn(lambda x:tf.matmul(x,self.weights)+self.biases,self.attention_z_reshape)

       self.pred=tf.nn.sigmoid(self.presoft)    
       if self.cost_type=='CE':
           self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.presoft,[-1,self.n_classes]), labels=tf.reshape(self.y_ph,[-1,self.n_classes])))              
       elif self.cost_type=='MI':
           self.cost=MI(self.pred,self.y_ph,1/4.,self.seq_len-2)
       elif self.cost_type=='CE-FL':
           self.cost=cross_entropy_focal_form(self.pred,self.y_ph)            
       elif self.cost_type=='MMD':
           self.cost=MMD(self.pred,self.y_ph,1/4.,self.seq_len-2)                   
       elif self.cost_type=='WMD':
           self.cost=WMD(self.pred,self.y_ph,1/4.,self.seq_len-2,1.0,cross_entropy_form) 
       elif self.cost_type=='WMD-FL':
           self.cost=WMD(self.pred,self.y_ph,1/4.,self.seq_len-2,1.0,cross_entropy_focal_form) 
               
       self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost) 

       self.init = tf.global_variables_initializer()
       self.saver = tf.train.Saver()
       self.saver_var = tf.train.Saver(tf.trainable_variables())
       if self.save_location==[]:
           self.save_location=os.getcwd()

     def train(self):
        
       self.iteration=0
       self.epoch=0
       self.prev_val_loss=100
       self.val_loss=99
       with tf.Session() as sess:
         sess.run(self.init)
         while self.epoch < self.minimum_epoch or self.prev_val_loss > self.val_loss:
             for i in range(self.num_batch):

                 sess.run(self.optimize, feed_dict={self.x_ph: np.reshape(np.expand_dims(self.features[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.input_feature_size]), self.y_ph: np.reshape(np.expand_dims(self.targ[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.n_classes]),self.dropout_ph:self.dropout, self.seq:np.ones(int(self.batch_size/self.snippet_length))*self.snippet_length, self.num_seqs:int(self.batch_size/self.snippet_length), self.seq_len:self.snippet_length})
                      
             print("Epoch " + str(self.epoch))
             if self.epoch > self.minimum_epoch:
                 self.loss_val=[]
                 for i in range(self.val_num_batch):
                     self.loss_val.append(sess.run(self.cost, feed_dict={self.x_ph: np.expand_dims(self.val[i*self.batch_size:(i+1)*self.batch_size,:],0), self.y_ph: np.expand_dims(self.val_targ[i*self.batch_size:(i+1)*self.batch_size,:],0),self.dropout_ph:1,self.seq:[self.batch_size],self.num_seqs:1, self.seq_len:self.batch_size}))
                     
                 self.prev_val_loss=self.val_loss
                 self.val_loss=np.mean(np.array(self.loss_val))              
                 print("Val Minibatch Loss " + "{:.6f}".format(self.val_loss))
             self.epoch+=1  
             if self.epoch==self.maximum_epoch:
                 break
         print("Optimization Finished!")
         self.saver.save(sess, self.save_location+'/'+self.filename)
         
     def implement(self,data):
             self.data=data
             self.test_out=[]
             with tf.Session() as sess:
                     self.saver.restore(sess, self.save_location+'/'+self.filename)
                     for i in range(len(self.data)):
                             self.test_out.append(sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(self.data[i],0),self.dropout_ph:1,self.seq:[len(self.data[i])],self.num_seqs:1, self.seq_len:len(self.data[i])})[0]) 

             return self.test_out
                  
##################################################################################
#             
## Example Implementation
             
# Toy problem trained on 10000 frames.
# val and test = 1000 frames
    
# load logarithmic spectrograms
TrainSpec=np.load('ENSTSpecTrain.npy')
TrainTarg=np.load('ENSTTargTrain.npy')
ValSpec=np.load('ENSTSpecVal.npy')
ValTarg=np.load('ENSTTargVal.npy')
TestSpec=np.load('ENSTSpecTrain.npy')[3000:5000]
TestTarg=np.load('ENSTTargTrain.npy')[3000:5000]

print(np.shape(TrainSpec))

# train the network and process test data
AFs=[]
AFs_train=[]
loss_functions=['CE-FL','WMD-FL']
#loss_functions=['CE','MI','MMD','WMD'] #change to this line to run all 4 versions
for c in loss_functions:
    print(c)
    NN=cnnSA3BF5(TrainSpec,TrainTarg, ValSpec, ValTarg,'PP_Example_'+c,minimum_epoch=100, maximum_epoch=200, n_hidden=[50,50], n_classes=3, attention_number=3, dropout=0.75,  learning_rate=0.003 ,save_location=[],snippet_length=100,cost_type='CE',batch_size=1000,input_feature_size=84*11,conv_filter_shapes=[[3,3,1,32],[3,3,32,64]], conv_strides=[[1,1,1,1],[1,1,1,1]], pool_window_sizes=[[1,3,3,1],[1,3,3,1]])
    NN.create()
    #NN.train()
    AFs.append(NN.implement([TestSpec])[0])

# plot the different activation functions for KD
np.save('AFs-CEflsa',AFs[0])
np.save('AFs-WMDflsa',AFs[1])


plt.subplot(3,1,1)
plt.plot(TestTarg[:,0])    
plt.subplot(3,1,2)
plt.plot(AFs[0][:,0])
plt.subplot(3,1,3)
plt.plot(AFs[1][:,0])
plt.show()
## plot all 4 loss functions
#
#plt.subplot(5,1,1)
#plt.plot(TestTarg[:,0])    
#plt.subplot(5,1,2)
#plt.plot(AFs[0][:,0])
#plt.subplot(5,1,3)
#plt.plot(AFs[1][:,0])
#plt.subplot(5,1,4)
#plt.plot(AFs[2][:,0])
#plt.subplot(5,1,5)
#plt.plot(AFs[3][:,0])


