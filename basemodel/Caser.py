# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:31:32 2018

@author: Yingpeng_Du
"""

from GCNdata import Data
import toolz
import numpy as np
import tensorflow as tf
from time import time
import argparse
import copy
from tqdm import tqdm 
import scipy.sparse as sp
from Train_module import Train_basic
def parse_args(name,factor,Topk,seed,batch_size):
    parser = argparse.ArgumentParser(description="Run .")  
    parser.add_argument('--name', nargs='?', default= name )    
    parser.add_argument('--model', nargs='?', default='Caser')
    parser.add_argument('--path', nargs='?', default='../datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 1e-6,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep', type=float, default=1.0, 
                    help='Keep probility (1-dropout) for the bilinear interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--Topk', type=int, default=Topk)
    parser.add_argument('--seed', type=int, default=seed)  
    return parser.parse_args()

class Caser(object):
    def __init__(self,args,data,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        # bind params to class
        self.args = args
        # bind params to class
        self.data = data
        self.num_users = self.data.entity_num['user']
        self.num_items = self.data.entity_num['item']
        self.learning_rate = learning_rate
        self.dims = hidden_factor
        self.l2 = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.L = 5
        self.T = 1
        self.n_v = 5
        self.n_h =16
        self.lengths  = [i + 1 for i in range(self.L)]
        self.drop_ratio = 0.5
        # init all variables in a tensorflow graph

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
            """
            """
            self.sequences = tf.placeholder(tf.int32, [None, self.L])
            self.users = tf.placeholder(tf.int32, [None, 1])
            self.items = tf.placeholder(tf.int32, [None, 2*self.T])
            self.is_training = tf.placeholder(tf.bool)
                                                 
            # user and item embeddings
            initializer = tf.contrib.layers.xavier_initializer()
            self.user_embeddings = tf.Variable(initializer([self.num_users, self.dims]))
            self.item_embeddings = tf.Variable(initializer([self.num_items, self.dims]))
            
            # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
            self.W2 = tf.Variable(initializer([self.num_items, self.dims+self.dims]))
            self.b2 = tf.Variable(initializer([self.num_items, 1]))
            
            item_embs = tf.nn.embedding_lookup(self.item_embeddings, self.sequences)
            item_embs = tf.reshape(item_embs, [-1, self.L, self.dims, 1])
            user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.users)
            user_emb = tf.reshape(user_emb, [-1, self.dims])
            
            # vertical convolution layers
            if self.n_v:
                out_v = tf.layers.conv2d(item_embs, 
                                         self.n_v, 
                                         [self.L, 1], 
                                         activation=tf.nn.relu)
                out_v = tf.contrib.layers.flatten(out_v)
                
            # horizontal convolution layers
            out_hs = list()
            if self.n_h:
                for h in self.lengths:
                    conv_out = tf.layers.conv2d(item_embs, 
                                                self.n_h, 
                                                [h, self.dims], 
                                                activation=tf.nn.relu)
                    conv_out = tf.reshape(conv_out, [-1, self.L-h+1, self.n_h])
                    pool_out = tf.layers.max_pooling1d(conv_out, [self.L-h+1], 1)
                    pool_out = tf.squeeze(pool_out, 1)
                    out_hs.append(pool_out)
                out_h = tf.concat(out_hs, 1)
                
            # concat two convolution layers    
            out = tf.concat([out_v, out_h], 1)
            
            # fully-connected layer
            z = tf.layers.dense(out, self.dims, activation=tf.nn.relu)
            z = tf.layers.dropout(z, self.drop_ratio, self.is_training)
            x = tf.concat([z, user_emb], 1)
            x = tf.reshape(x, [-1, 1, 2*self.dims])
            
            w2 = tf.nn.embedding_lookup(self.W2, self.items)
            b2 = tf.nn.embedding_lookup(self.b2, self.items)
            b2 = tf.squeeze(b2, 2)
            
            # training with negative samples
            pred = tf.squeeze(tf.matmul(x, tf.transpose(w2, perm=[0,2,1])), 1) + b2        
            self.target_pred, negative_pred = tf.split(pred, 2, axis=1)
        
            # loss
            positive_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.target_pred)))
            negative_loss = -tf.reduce_mean(tf.log(1 - tf.nn.sigmoid(negative_pred)))
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2
    
            self.loss = positive_loss + negative_loss + l2_loss
            
            # optimizer
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
            # For test
            self.all_items = tf.placeholder(tf.int32, [None, self.num_items])        
            test_w2 = tf.nn.embedding_lookup(self.W2, self.all_items)
            test_b2 = tf.nn.embedding_lookup(self.b2, self.all_items)        
            test_b2 = tf.reshape(test_b2, [-1, self.num_items])
            self.test_pred = tf.reduce_sum(tf.multiply(x, test_w2), axis=2) + test_b2
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)
            

            self.out_all_topk = tf.nn.top_k(self.test_pred,200)
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
    def partial_fit(self, data):  # fit a batch        
        feed_dict = {self.users: data['users'],self.items:data['items'],self.sequences:data['seq'],self.is_training:True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return [loss]
    def pairwise_loss(self,inputx,labels):

        inputx_f = inputx[1:]
        inputx_f = tf.concat([inputx_f,tf.zeros([1,1])],axis=0)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid((inputx-inputx_f)*labels)))
        return loss
    def pointwise_loss(self,inputx,labels):
        return tf.nn.l2_loss(inputx-labels)
    def cross_entropy_loss(self,inputx,labels):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=inputx, labels=labels))

    def topk(self,user_item_feedback):
        user =np.expand_dims( user_item_feedback[:,0],axis=1)
        items = np.tile([[i for i in range(self.num_items)]],[len(user_item_feedback),1])
        seq = user_item_feedback[:,2:]
        feed_dict ={ self.users: user,self.all_items:items,self.sequences:seq,self.is_training:False}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)     
        return self.prediction
    

class Train(Train_basic):
    def __init__(self,args,data):
        super(Train,self).__init__(args,data)
        self.model = Caser(self.args,self.data ,args.hidden_factor,args.lr, args.lamda, args.optimizer)
    def train(self):  # fit a dataset
        MAP_valid = 0
        if self.include_valid ==True:
            train_data = np.array(self.data.train) # Array形式的二元组（user,item），none * 2
            basemodel = 'basemodel_v'
        else:
            train_data = np.array(self.data.train + self.data.valid) # Array形式的二元组（user,item），none * 2
            basemodel = 'basemodel'

        for epoch in tqdm(range(0,self.epoch+1)): #每一次迭代训练
            np.random.shuffle(train_data)

            #sample负样本采样
            NG = 1#NG倍举例
            NegSample = self.sample_negative(train_data,NG)#采样，none * NG
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(train_data))] ):                
                chunk = list(user_chunk)
                neg_chunk = np.array(NegSample[chunk],dtype = np.int)[:,0]#none
                train_chunk  = train_data[chunk]#  none * 2
                train_chunk_p5 = np.concatenate([train_chunk,np.array([\
                    self.data.latest_interaction[(line[0],line[1])] for line in train_chunk])],axis =1)#none*2+5
                users = np.expand_dims(train_chunk_p5[:,0] ,axis=1)           
                items = np.stack([train_chunk_p5[:,1],neg_chunk],axis=1)
                sequence = train_chunk_p5[:,2:]
                self.feedback = {'users':users,'items':items,'seq':sequence}
                #meta-path feature
                loss =  self.model.partial_fit(self.feedback)
            t2 = time()

         # evaluate training and validation datasets
            if epoch % int(self.args.epoch/10) == 0:
                for topk in [10]:
                    init_test_TopK_test = self.evaluate_TopK(self.data.test,topk) 
                    print("Epoch %d Top%d \t TEST SET:%.4f MAP:%.4f,NDCG:%.4f,PREC:%.4f;[%.1f s]\n"
                      %(epoch,topk,0,init_test_TopK_test[0],init_test_TopK_test[1],init_test_TopK_test[2], time()-t2))
                if MAP_valid < np.sum(init_test_TopK_test) and epoch<self.epoch:
                    MAP_valid = np.sum(init_test_TopK_test)
                    self.meta_result = self.save_meta_result()
                else:
                    
                    np.save("../datasets/%s/%s/%s.npy"%(basemodel,self.args.name,self.args.model),self.meta_result )
                    break      
def Caser_main(name,factor,seed,batch_size):    

#name,factor,Topk,seed ,batch_size = 'CiaoDVD',64,10,0,2056
    args = parse_args(name,factor,0,seed,batch_size)
    data = Data(args,seed)#获取数据
    session_DHRec = Train(args,data)
    session_DHRec.train()
    # 
