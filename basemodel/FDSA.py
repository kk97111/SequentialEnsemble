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
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from Train_module import Train_basic


NUM = 3
def parse_args(name,factor,seed,batch_size):  
    parser = argparse.ArgumentParser(description="Run .")  
    parser.add_argument('--name', nargs='?', default= name )    
    parser.add_argument('--model', nargs='?', default='FDSA')
    parser.add_argument('--path', nargs='?', default='../datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
#    parser.add_argument('--epoch', type=int, default=100,#ciaoDVD 600 #Amazon_App 200 ML&dianping 100
#                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 10e-3,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--seed', type=int, default=seed)  
    parser.add_argument('--epoch', type=int, default=100,help='Number of epochs.')
    return parser.parse_args()

class FDSA(object):
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
        self.n_attribute = len(self.data.item_side_entity)
        self.n_slot = self.n_attribute + 1
        # init all variables in a tensorflow graph
        np.random.seed(args.seed)
        self.num_a = np.sum([self.data.entity_num[key] for key in self.data.item_side_entity])
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
            self.weights = self._initialize_weights()

            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.labels = tf.placeholder(tf.float32, shape=[None,1])# none 
            self.neg_items = tf.placeholder(tf.int32, shape=[None, 5])
            self.all_attributes = tf.placeholder(tf.int32, shape=[self.n_item,self.n_attribute,NUM])


            self.users_idx = self.feedback[:,0]#none
            self.items_idx = self.feedback[:,1]#none
            self.target_items = tf.concat([self.feedback[:,3:],tf.expand_dims(self.items_idx,-1)],axis=-1)#none * 5
            
            self.item_sequence = self.feedback[:,2:] # none * 5
            
            feature_embs =  tf.reduce_mean(tf.nn.embedding_lookup(self.weights['attribute_embeddings'],self.all_attributes),axis=2)#all*num_attri*k
          #pair interaction
            self.item_sequence_embs =  tf.nn.embedding_lookup(self.weights['sequence_embeddings'],self.item_sequence)#none *5* d
            self.feature_sequence_embs = tf.nn.embedding_lookup(feature_embs,self.item_sequence)#none *5 * feature * d
            att_feature   =   tf.nn.softmax(tf.layers.dense(self.feature_sequence_embs,self.hidden_factor),axis=2)
            self.feature_sequence_embs = tf.reduce_sum(self.feature_sequence_embs * att_feature,axis=2)#none *5* d

            F = self.feature_sequence_embs + self.weights['position']#none *5* d
            S =self.item_sequence_embs #+ self.weights['position']#none *5* d
            self.O_s =self.SAB( S,'item')#none *5* d
            self.O_f =self.SAB( F,'feature')#none *5* d
            
            O_sf = tf.concat([self.O_s,self.O_f],axis=-1)#none *5* 2d
            
            O_sf = tf.layers.dense(O_sf,self.hidden_factor) #none *5* d        
            
            O_sft =tf.expand_dims( O_sf[:,-1,:],axis=1)#none*1 * d   
            
            self.item_pos =  tf.nn.embedding_lookup(self.weights['item_embeddings'],self.target_items)#none *5* d
            self.item_neg =  tf.nn.embedding_lookup(self.weights['item_embeddings'],self.neg_items)#none *5*d
            
            self.out_pos =  tf.reduce_sum( O_sft * self.item_pos,axis=-1,keep_dims=True)#none * 1  
            self.out_neg =  tf.reduce_sum( O_sft * self.item_neg,axis=-1,keep_dims=True)#none * 1  
            
            
            self.loss_rec = -tf.reduce_sum(tf.log(tf.sigmoid(self.out_pos))+tf.log(tf.sigmoid(1- self.out_neg)))

#            self.loss_rec = self.pairwise_loss(self.out,self.labels)\
#                            + self.pairwise_loss(self.out,self.labels)
            
            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)      


            self.loss = self.loss_rec+ self.loss_reg 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        


            
            out = tf.matmul(O_sf[:,-1,:],self.weights['item_embeddings'],transpose_b=True)
            self.out_all_topk = tf.nn.top_k(out,200)
            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)


    #For model

    def jieduan(self,x,name):
        return tf.layers.dense(x,self.hidden_factor,use_bias=False)
    def scaled_dot_product_attention(self, queries, keys, values,name):
        self.num_heads = 4
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        Q, K, V = self.jieduan(queries,name), self.jieduan(keys,name), self.jieduan(values,name)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, int(self.hidden_factor/self.num_heads)]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, int(self.hidden_factor/self.num_heads)]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, int(self.hidden_factor/self.num_heads)]), [0, 2, 1, 3])
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_factor))), V)
        S = tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [batch_size, num_queries, int(self.hidden_factor)])
        return S#tf.keras.layers.LayerNormalization(axis=-1)(S)#tf.stop_gradient(S)* tf.Variable(np.ones([1,1,self.hidden_factor]),dtype=tf.float32)
    def SAB(self,sequence_embs,name):
        #sequence_embs #none *5* d

        M_f = self.scaled_dot_product_attention(sequence_embs,sequence_embs,sequence_embs,name)
        LayerNormalization = tf.keras.layers.LayerNormalization()
        M_f = LayerNormalization(M_f+sequence_embs)
        O_f = tf.layers.dense(tf.layers.dense(M_f,self.hidden_factor),self.hidden_factor,activation=tf.nn.relu)
        O_f = LayerNormalization(O_f+sequence_embs)
        return O_f          
    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks
    
    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return tf.Session(config=config)
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['attribute_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_attribute, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['sequence_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['position']  =  tf.Variable(np.random.normal(0.0, 0.001,[1,5, self.hidden_factor]),dtype = tf.float32) #1 *5* d

        return all_weights


    def partial_fit(self, data):  # fit a batch
        neg = np.random.randint(0,self.n_item,np.shape(data['feedback'][:,2:]))
        feed_dict = {self.feedback: data['feedback'],self.all_attributes:data['all_attributes'],self.neg_items:neg}
        loss_rec,loss_reg, opt = self.sess.run((self.loss_rec,self.loss_reg, self.optimizer), feed_dict=feed_dict)
        return loss_rec,0,loss_reg

    def topk(self,user_item_feedback,all_attributes):
        
        feed_dict = {self.feedback: user_item_feedback,self.all_attributes:all_attributes}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)     
        return self.prediction
    def pairwise_loss(self,inputx,labels):
#        input none*1
#        label none*1
        A = - tf.log(tf.sigmoid(inputx) * (labels*2-1) + tf.abs(1-labels)) 
        
        return tf.reduce_sum(A)      #tf.nn.l2_loss(inputx-labels)

class Train(Train_basic):
    def __init__(self,args,data):
        super(Train,self).__init__(args,data)
        self.item_attributes = self.collect_attributes()
        self.model = FDSA(self.args,self.data ,args.hidden_factor,args.lr, args.lamda, args.optimizer)
    def sample_negative(self, data,num=10):
        samples = np.random.randint( 0,self.n_item,size = (len(data)))
        return samples


def FDSA_main(name,factor,seed,batch_size):    
#name,factor,Topk,seed ,batch_size = 'CiaoDVD',64,10,0,1024
    args = parse_args(name,factor,seed,batch_size)
    data = Data(args,seed)#获取数据
    session_DHRec = Train(args,data)
    session_DHRec.train_attribute()
    # 