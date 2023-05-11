# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:31:32 2018

@author: Yingpeng_Du
"""

from GCNdata import Data
import numpy as np
import tensorflow as tf
import argparse
from Train_module import Train_basic
#n=1
def parse_args(name,factor,seed,batch_size,n=5):
    parser = argparse.ArgumentParser(description="Run .")  
    parser.add_argument('--name', nargs='?', default= name )    
    parser.add_argument('--model', nargs='?', default='PFMC')
    parser.add_argument('--path', nargs='?', default='../datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 10e-5,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--seed', type=int, default=seed) 
    parser.add_argument('--n', type=int, default=n) 
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    return parser.parse_args()

class PFMC(object):
    def __init__(self,args,data,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        # bind params to class
        self.args = args
        # bind params to class
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.loss_type = 'square_loss'

        # init all variables in a tensorflow graph
        np.random.seed(args.seed)

        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        tf.reset_default_graph()  # 重置默认图       
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            # Input data.

            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.labels = tf.placeholder(tf.float32, shape=[None,1])# none 
            
            self.weights = self._initialize_weights()

            self.users_idx = self.feedback[:,0]#none
            self.items_idx = self.feedback[:,1]#none
            self.users_p5_idx = self.feedback[:,7-self.args.n:7] # none * 5
            self.l1_loss = []
            #self.embeddings = self.batch_embeddings(self.weights)
            self.UI = tf.nn.embedding_lookup(self.weights['UI'],self.users_idx)#none*k
            self.IU = tf.nn.embedding_lookup(self.weights['IU'],self.items_idx)#none*k
            self.out1 = tf.reduce_sum(self.UI * self.IU,axis=1,keep_dims=True)
            
            self.IL = tf.reduce_mean(tf.nn.embedding_lookup(self.weights['IL'],self.users_p5_idx),axis=1)#none*k
            self.LI = tf.nn.embedding_lookup(self.weights['LI'],self.items_idx)#none*k
            self.out2 = tf.reduce_sum(self.IL * self.LI,axis=1,keep_dims=True)
            
#            self.UL = tf.nn.embedding_lookup(self.weights['UI'],self.users_idx)#none*k
#            self.LU = tf.reduce_mean(tf.nn.embedding_lookup(self.weights['LU'],self.users_p5_idx),axis=1)#none*k
#            
            
            
            self.out = self.out1 + self.out2#none * 1            
            self.loss_rec = self.pairwise_loss(self.out,self.labels)
#            self.loss_l1 = tf.reduce_sum(tf.stack(self.l1_loss))tf.Variable(0,dtype=tf.float32)#
            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg +=self.args.lamda* tf.nn.l2_loss(wgt)
            self.loss = self.loss_rec +self.loss_reg
            
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)
            
            self.score1= tf.matmul(self.UI,tf.transpose(self.weights['IU']))
            self.score2 = tf.matmul(self.IL,tf.transpose(self.weights['LI']))
            self.score = self.score1 + self.score2
            self.out_all_topk = tf.nn.top_k(self.score,200)
            self.trainable = tf.trainable_variables()
            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    #For model
    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        return tf.Session(config=config)
    
    def _initialize_weights(self):
        all_weights = dict()
        for key in self.data.entity:
            n_entity = self.data.entity_num[key]
            all_weights[key+'_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[n_entity, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['UI'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_user, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['IU'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['IL'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['LI'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['UL'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_user, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['LU'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K

        return all_weights

    def partial_fit(self, data):  # fit a batch
        
        feed_dict = {self.feedback: data['feedback'],self.labels:data['labels']}
        loss_rec,loss_reg, opt = self.sess.run((self.loss_rec,self.loss_reg, self.optimizer), feed_dict=feed_dict)
        return loss_rec,loss_reg
    def pairwise_loss(self,inputx,labels):
#        input none*1
#        label none*1
        inputx_f = inputx[1:]
        inputx_f = tf.concat([inputx_f,tf.zeros([1,1])],axis=0)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid((inputx-inputx_f)*labels)))
        return loss
        
    def topk(self,user_item_feedback):
        feed_dict = {self.feedback: user_item_feedback}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)     
        return self.prediction

class Train(Train_basic):
    def __init__(self,args,data):
        super(Train,self).__init__(args,data)
        self.model = PFMC(self.args,self.data ,args.hidden_factor,args.lr, args.lamda, args.optimizer)


def PFMC_main(name,factor,seed,batch_size,keep):   

#name,factor,Topk,seed ,batch_size = 'ML',64,10,0,2048
    args = parse_args(name,factor,seed,batch_size,keep)
    data = Data(args,seed)#获取数据
    session_DHRec = Train(args,data)
    session_DHRec.train()
