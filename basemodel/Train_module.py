# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:50:04 2022

@author: dyp
"""
import os
import numpy as np
import copy
from tqdm import tqdm
import toolz
from time import time
include_valid = True

class Train_basic(object):
    def __init__(self,args,data):
        self.args = args
        self.data = data
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.entity = self.data.entity
        self.user_side_entity = self.data.user_side_entity
        self.item_side_entity = self.data.item_side_entity          
        # Data loadin
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.include_valid = include_valid
    def train(self):  # fit a dataset
        # Check Init performance
        #初始结果   
        MAP_valid = 0
        if include_valid ==True:
            PosSample = np.array(self.data.train) # Array形式的二元组（user,item），none * 2
            basemodel = 'basemodel_v'
        else:
            basemodel = 'basemodel'
            PosSample = np.array(self.data.train + self.data.valid) # Array形式的二元组（user,item），none * 2

        PosSample_with_p5 = np.concatenate([PosSample,np.array([\
                    self.data.latest_interaction[(line[0],line[1])] for line in PosSample])],axis =1)#none*2+5
        for epoch in tqdm(range(0,self.epoch+1)): #每一次迭代训练
            np.random.shuffle(PosSample)
            #sample负样本采样
            NG = 1#NG倍举例
            NegSample = self.sample_negative(PosSample,NG)#采样，none * NG
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(PosSample))] ):                
                chunk = list(user_chunk)
                neg_chunk = np.array(NegSample[chunk],dtype = np.int)[:,0]#none*1
                train_chunk_p5 = PosSample_with_p5[chunk]#none*2+5
                train_chunk_p5_copy = copy.deepcopy(train_chunk_p5)            
                train_chunk_p5_copy[:,1] = neg_chunk
                    
                feedback = np.stack([train_chunk_p5,train_chunk_p5_copy],axis=1)
                feedback = np.reshape(feedback,[-1,2+5])
                labels = np.reshape(np.stack([np.ones(len(chunk)),np.zeros(len(chunk))],axis=1),[-1,1])
                #meta-path feature
                self.feed_dict = {'feedback':feedback,'labels':labels}
                loss =  self.model.partial_fit(self.feed_dict)
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
                    dir_name = "../datasets/%s/%s"%(basemodel,self.args.name)
                    print([dir_name,os.path.isdir(dir_name)])
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    np.save("../datasets/%s/%s/%s.npy"%(basemodel,self.args.name,self.args.model),self.meta_result )
                    break
    def train_attribute(self):  # fit a dataset
        MAP_valid = 0
        if include_valid ==True:
            PosSample = np.array(self.data.train) # Array形式的二元组（user,item），none * 2
            basemodel = 'basemodel_v'
        else:
            basemodel = 'basemodel'
        PosSample_with_p5 = np.concatenate([PosSample,np.array([\
                    self.data.latest_interaction[(line[0],line[1])] for line in PosSample])],axis =1)#none*2+5
        for epoch in tqdm(range(0,self.epoch+1)): #每一次迭代训练
            np.random.shuffle(PosSample)
            #sample负样本采样
            np.random.shuffle(PosSample)
            #sample负样本采样
            NG = 1#NG倍举例
            NegSample = self.sample_negative(PosSample,NG)#采样，none * NG
            for user_chunk in toolz.partition_all(self.batch_size,[i for i in range(len(PosSample))] ):                
                chunk = list(user_chunk)
                neg_chunk = np.array(NegSample[chunk],dtype = np.int)#none*1
                train_chunk_p5 = PosSample_with_p5[chunk]#none*2+5
                train_chunk_p5_copy = copy.deepcopy(train_chunk_p5)            
                train_chunk_p5_copy[:,1] = neg_chunk
                    
                feedback = np.stack([train_chunk_p5,train_chunk_p5_copy],axis=1)
                feedback = np.reshape(feedback,[-1,2+5])
                labels = np.reshape(np.stack([np.ones(len(chunk)),np.zeros(len(chunk))],axis=1),[-1,1])
                #meta-path feature
                self.feed_dict = {'feedback':feedback,'labels':labels,'all_attributes':self.item_attributes}
                loss =  self.model.partial_fit(self.feed_dict)
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
                    dir_name = "../datasets/%s/%s"%(basemodel,self.args.name)
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    np.save("../datasets/%s/%s/%s.npy"%(basemodel,self.args.name,self.args.model),self.meta_result )
                    break                

    def save_meta_result(self):
        #inputs:
        candidate = copy.deepcopy(self.data.dict_list['user_item'])
        size = len(candidate)
        num = 200

        last_iteraction = []
        for line in candidate:
            #meta-path特征
            user,item = line
            last_iteraction.append(self.data.latest_interaction[(user,item)])        
        last_iteraction = np.array(last_iteraction)
        score = []
        for _ in range(int(size/num+1)):
            user_item_block = candidate[_*num:(_+1)*num]
            last_iteraction_block = last_iteraction[_*num:(_+1)*num]
            feedback_block = np.concatenate((user_item_block,last_iteraction_block),axis=1)
            try:
                score_block = self.model.topk(feedback_block,self.item_attributes)                
            except:
                score_block = self.model.topk(feedback_block)
            
            score_block = score_block[:,:100].tolist()
            score.extend(score_block) 
        score = np.array(score,dtype=np.int)
        return np.concatenate((candidate,score),axis=1)
        
            
        
    def sample_negative(self, data,num=10):
        samples = np.random.randint( 0,self.n_item,size = (len(data),num))

        return samples
    def collect_attributes(self):
        #return item * 
        NUM = 3
        attributes = []
        start_index = 0
        for entity in self.data.item_side_entity:
            key = 'item_' + entity
            attribute_item = self.data.dict_forward[key]
            attribute = []
            for item in range(self.n_item):
                list_ = attribute_item[item]
                if len(list_) <=NUM:
                    attribute.append(list_+[-1 for i in range(NUM-len(list_))])
                else:
                    attribute.append(list_[:NUM])
            attribute = np.array(attribute)
            attributes.append(attribute + start_index)
            start_index = start_index + self.data.entity_num[entity]
        return np.stack(attributes,axis=1)

    def evaluate_TopK(self,test,topk):
        test_candidate = copy.deepcopy(np.array(test))#none * 2
        size = len(test_candidate)
        result_MAP = []
        result_PREC = []
        result_NDCG = []
        num = 100
        #meta-path feature
        last_iteraction = [] #none*5
        for line in test_candidate:
            #meta-path特征
            user,item = line
            last_iteraction.append(self.data.latest_interaction[(user,item)])
            
        last_iteraction = np.array(last_iteraction)
        for _ in range(int(size/num+1)):
            user_item_block = test_candidate[_*num:(_+1)*num]
            last_iteraction_block = last_iteraction[_*num:(_+1)*num]
            feedback_block = np.concatenate((user_item_block,last_iteraction_block),axis=1)
            try: 
                prediction= self.model.topk(feedback_block,self.item_attributes) #none * 50
            except:
                prediction= self.model.topk(feedback_block) #none * 50

                
            assert len(prediction) == len(feedback_block)
#            print(_)
            for i,line in enumerate(user_item_block):
                user,item = line
                n = 0 
#                print(prediction[i])
                for it in prediction[i]:
                    if n> topk -1:
                        result_MAP.append(0.0)
                        result_NDCG.append(0.0)
                        result_PREC.append(0.0)  
                        n=0
                        break
                    elif it == item:   
#                        print([it,item])
                        result_MAP.append(1.0)
                        result_NDCG.append(np.log(2)/np.log(n+2))
                        result_PREC.append(1/(n+1))
                        n=0
                        break
                    elif it in self.data.set_forward['train'][user] or it in self.data.set_forward['valid'][user]:
                        continue
                    else:
                        n = n + 1   
        print(np.sum(result_MAP))
        return  [np.mean(result_MAP),np.mean(result_NDCG),np.mean(result_PREC)] 