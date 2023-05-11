import copy
import numpy as np
import pandas as pd 
from collections import Counter
from scipy.sparse import csr_matrix, coo_matrix,lil_matrix, save_npz
import codecs
from tqdm import tqdm
leave_num = 1000000000
remove_rating = 3.5
last = 5# 5for BMS;10for SNR

def relation_dict(n1,n2,list1):
    #将总数n1，n2的entity转化成映射字典
    dict_forward = {i:[] for i in range(n1)}
    dict_reverse = {i:[] for i in range(n2)}
    for x,y in list1:
        if y not in dict_forward[x]:
            dict_forward[int(x)].append(int(y))
            dict_reverse[int(y)].append(int(x))
    return dict_forward
def remove_unrating_user_and_rename(tranA,tranB,list1):
    #去掉没有的user对应的rating，并rename
    res = []
    for x,y in list1:
        if x in tranA.keys() and y in tranB.keys():      
            res.append([tranA[x],tranB[y]])
    return res

def reverse_and_map(l):
    #做rename，???->n
    return {v: k for k, v in enumerate(l)}

def set_forward(dict1):
    return_1 = dict()
    for key in dict1.keys():
        sub = dict1[key]
        return_1[key] = { ky: set(sub[ky]) for ky in sub.keys()}
    return return_1

def build_sparse_matrix(n_1,n_2,list1):
    res = lil_matrix((n_1,n_2))
    for user in list1.keys():
        for item in list1[user]:
            res[user,item] =1
    return res    
    
    
    

class Data(object):
    def __init__(self,args,seed=0,Markov = False):
        self.name_id = dict() 
        self.name = args.dataset
        self.dir = args.path if args.path[-1] == '/' else args.path + '/'
        if args.name in ['CiaoDVD','dianping']:
            self.encoding = 'utf-8'
        else:
            self.encoding= 'iso-8859-15'

#        if self.name in ['ML']:
#            self.entity = ['user','item','G','A','C','D']#
#            self.user_side_entity = []
#            self.item_side_entity = ['G','A','C','D']#
#            self.drop_user,self.drop_item = 10,10
#        if self.name in ['CiaoDVD']:
#            self.entity = ['user','item','G']#'C',
#            self.user_side_entity = []
#            self.item_side_entity = ['G']#'C'
#            self.drop_user,self.drop_item = 3,3
        if self.name in ['Amazon_App','Games','ml1m']:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user,self.drop_item = 10,10
        if self.name in ['Grocery']:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user,self.drop_item = 20,20          
        if self.name in ['Sport','Clothing','Instant_Video','Pet_Supplies','Patio','Office_Products','Musical_Instruments','Digital_Music','Baby','Automotive']:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user,self.drop_item = 5,5
        if self.name in ['dianping']:
            self.entity = ['user','item','S','A','C']#
            self.user_side_entity = []
            self.item_side_entity = ['S','A','C']#'C'
            self.drop_user,self.drop_item = 20,20#之前是25,50
        if self.name in ['Kindle','tiktok']:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user,self.drop_item = 50,50#之前是50,50

        
       #save 
        self.dict_entity2id = dict()#entity对应的ID
        self.dict_list = dict()#entity1_entity2对应的两元组
        self.dict_forward = dict()#dict entity1_entity2对应的关系字典
        self.dict_reverse = dict()
        self.entity_num = dict()#数量
        # read rating and split train and test set     
        #Remove user，item，which their number is less than 5 interaction。
        self.dict_list['user_item'],self.entity_num['user'], self.entity_num['item'],\
        self.dict_entity2id['user'], self.dict_entity2id['item'] = self._get_rating(remove_rating,self.drop_user,self.drop_item)#53
        self.latest_interaction = self.find_latest_interaction(self.dict_list['user_item']) #时间特征

        self.train, self.valid,self.test = self.split_traintest(self.dict_list['user_item'])
        
        #Time Sequence Feature

        
        
        self.dict_forward['train'] = relation_dict(self.entity_num['user'],self.entity_num['item'],self.train)
        self.dict_forward['valid'] = relation_dict(self.entity_num['user'],self.entity_num['item'],self.valid)
        self.dict_forward['test'] = relation_dict(self.entity_num['user'],self.entity_num['item'],self.test)
#        self.dict_forward['valid']= relation_dict(self.entity_num['user'],self.entity_num['item'],self.valid)

        self.markov = self.constrcut_markov_matrices()
        
        
#        business to others and set thier forward and reverse dictionary


#        if self.name in ['ML','CiaoDVD','Amazon_App','dianping']: 
        for entity in self.item_side_entity:
            self.dict_list['item_'+entity],self.dict_entity2id[entity],self.entity_num[entity] = self.get_b_other('I'+entity+'.data')
            self.dict_forward['item_'+entity] = relation_dict(self.entity_num['item'],self.entity_num[entity], self.dict_list['item_'+entity])
        for entity in self.user_side_entity:
            self.dict_list['user_'+entity],self.dict_entity2id[entity],self.entity_num[entity] = self.get_u_other('U'+entity+'.data')
            self.dict_forward['user_'+entity] = relation_dict(self.entity_num['user'],self.entity_num[entity], self.dict_list['user_'+entity])

#       build sparse matrice of entity-entity
        #user -item
        self.matrix = dict()
        self.matrix['user_item']  =  build_sparse_matrix(self.entity_num['user'],self.entity_num['item'],self.dict_forward['train'])
        self.matrix['item_user']  =  self.matrix['user_item'].transpose()

        #user - ?
        for entity in self.user_side_entity:
            self.matrix['user'+entity]  =  build_sparse_matrix(self.entity_num['user'],self.entity_num[entity],self.dict_forward['user_'+entity])
            self.matrix[entity+'user']  =  self.matrix['user'+entity].transpose()
        #item - ?
        for entity in self.item_side_entity:
            self.matrix['item'+entity]  =  build_sparse_matrix(self.entity_num['item'],self.entity_num[entity],self.dict_forward['item_'+entity])
            self.matrix[entity+'item']  =  self.matrix['item'+entity].transpose()
#        self.set_forward = dict()
#        for key in self.dict_forward.keys():
        self.set_forward = set_forward(self.dict_forward)
        print('user:%d\t item:%d\t train:%d\t'%(self.entity_num['user'], self.entity_num['item'],len(self.train)))
        if self.name in ['Grocery','Kindle','Games']:
            self.pic = self.pic_feature('feature.npy')
        if self.name in ['tiktok']:
            self.pic = self.pic_feature('visual.npy')        
            self.acou = self.pic_feature('acoustic.npy')  
#    def split_traintest(self,u_i,ratio):
##        ratio = 0.9 #Must note
#        user_count = Counter(u_i[:,0]) #count number of user's interaction
#        train = []
#        test = []
#        valid = []
#        self.test_one_for_all = []
#        self.valid_one_for_all = []
#        fisrt_valid = True 
#        fisrt_test = True
#        n = 0
#        for i,user_item in enumerate(u_i):
#            user,item = user_item
#            if n < min(int(user_count[user]*0.95),user_count[user]-2): #
#                train.append([user,item])
#                n = n + 1
#            elif min(int(user_count[user]*0.95),user_count[user]-1):#n < min(int(user_count[user]*0.95),
#                valid.append([user,item])
#                if fisrt_valid == True: 
#                    sequence = copy.deepcopy(self.latest_interaction[(user,item)])
#                    sequence.insert(0,0)                    
#                    sequence.insert(0,user)
#                    list_uq = sequence
#                    self.valid_one_for_all.append(list_uq)
#                    fisrt_valid = False
#                n = n + 1
#            else:
#                test.append([user,item])
#                if fisrt_test == True:     
#                    sequence = copy.deepcopy(self.latest_interaction[(user,item)])
#                    sequence.insert(0,0)                    
#                    sequence.insert(0,user)
#                    list_uq = sequence
#                    self.test_one_for_all.append(list_uq)
#                    fisrt_test = False
#            try:
#                if u_i[i+1][0] != user:
#                    n = 0
#                    fisrt_valid = True 
#                    fisrt_test = True                    
#            except:
#                pass
#        return train,test,valid
    def split_traintest(self,u_i,ratio = [0.6,0.8]):
        #保持6:2:2和JMLR中的一致
        user_count = Counter(u_i[:,0]) #count number of user's interaction
        train = []
        valid = []
        test = []
        n = 0
        for i,user_item in enumerate(u_i):
            user,item = user_item
            if n <  min(int(user_count[user]*ratio[0]),user_count[user]-4): #
                train.append([user,item])
                n = n + 1
            # leave one out:user_count[user]-1:
            # leave percent out: min(int(user_count[user]*ratio[1]),user_count[user]-1):
            elif n < user_count[user]-1:
                valid.append([user,item])
                n = n + 1
            else:
                test.append([user,item])
            try:
                if u_i[i+1][0] != user:
                    n = 0                  
            except:
                pass
        return train,valid,test

    def find_latest_interaction(self,u_i,keep = last):
        result = dict()
        init = [-1 for i in range(keep)]
        latest = copy.deepcopy(init)
        for i,user_item in enumerate(u_i):
            user,item = user_item
            result[(user,item)] =  copy.deepcopy(list(latest[-keep:]))
            if i<len(u_i)-1:
                if u_i[i+1][0] != user:
                    latest = copy.deepcopy(init)
                else:
                    latest.append(item)
                
        return result
    def _get_rating(self,score,remove_less_than1,remove_less_than2,keep_interaction = -1):
        file_path = self.dir + 'ratings.data'
        df = pd.read_csv(file_path,sep='\t',header = None,nrows =leave_num)
        df.columns = ['user','item','rating','time']
        df = df.sort_values(['user','time']) # sort for user and time
        #remv ratings
        df = df[df['rating']>score]
        
        u_i = df[['user','item']].values.tolist()
        u_i = [list(map(str,line)) for line in u_i]
        #re
        
        us,bs = zip(*u_i)
        us_list = []
        bs_list = []
        C_us = Counter(us)
        for u in C_us.keys():
            if C_us[u]>=remove_less_than1:
                us_list.append(u)
        C_bs = Counter(bs)
        for b in C_bs.keys():
            if C_bs[b]>=remove_less_than2:
                bs_list.append(b)                        
        num_user, num_item = len(us_list), len(bs_list)
        user2id, item2id = reverse_and_map(us_list), reverse_and_map(bs_list)
        self.name_id['item'] = item2id
        return np.array(remove_unrating_user_and_rename(user2id,item2id, u_i)),num_user, num_item,user2id, item2id
    
    def get_b_other(self,subdir):
        b_other = []
        file_path = self.dir + subdir
        print(file_path)
        with codecs.open(file_path,'r',encoding=self.encoding) as rfile:#iso-8859-15
            for line in rfile:
#                if self.name in ['ml100k']:
#                    line = line.strip().split(',')   
#                if self.name in ['ML','tiktok' ,'CiaoDVD','Games','Amazon_App','dianping','Grocery','Kindle']:
                line = line.strip().split('\t')  
                if (len(line)!=2):
                    print(line)
                b_other.append([str(line[0]),str(line[1])])
        bs,others = zip(*b_other)           
        others = list(set(others))    
        others2id = reverse_and_map(others)
        self.name_id[subdir] = others2id
        return remove_unrating_user_and_rename(self.dict_entity2id['item'],others2id,b_other) , others2id,len(others)    
    def get_u_other(self,subdir):
        b_other = []
        file_path = self.dir + subdir
        with open(file_path) as rfile:
            for line in rfile:
                if self.name=='yelp50k' or self.name=='yelp200k' or self.name=='doubanDVD':
                    line = line.strip().split()
                if self.name=='amazon':
                    line = line.strip().split(',')    
                if self.name=='ml100k':
                    line = line.strip().split(',')   
                b_other.append(line)
        bs,others = zip(*b_other)           
        others = list(set(others))    
        others2id = reverse_and_map(others)
        return remove_unrating_user_and_rename(self.dict_entity2id['user'],others2id,b_other) , others2id,len(others)    
    def constrcut_markov_matrices(self,keep = 1):
#        markov = {i:lil_matrix((self.entity_num['item'],self.entity_num['item'])) for i in range(keep)}
        markov = lil_matrix((self.entity_num['item'],self.entity_num['item']))
        
        for user,item in tqdm(self.train,'markov_preparing'):
             item_ps = self.latest_interaction[(user,item)]
             for item_p in item_ps[-keep:]:
                 if item_p >= 0:
                     markov[item_p,item] +=1
        return markov
    def pic_feature(self,name,dims=64):
        pic_feature = np.zeros([self.entity_num['item'],dims])
        #read feature
        file_path = self.dir + name#'feature.npy'
        feature = np.load(file_path)
        #read items rank
        with codecs.open(self.dir + 'feature.data','r',encoding=self.encoding) as rfile:#iso-8859-15
            for line in rfile:
                line = line.strip().split('\t') 
                item = line[0]
                idx =int(line[1])
                if item in self.dict_entity2id['item']:
                    pic_feature[self.dict_entity2id['item'][item]] = feature[idx]
        return pic_feature
        
#                if self.name in ['ml100k']:
#                    line = line.strip().split(',')   
#                if self.name in ['ML', 'CiaoDVD','Amazon_App','dianping']:
#                    line = line.strip().split('\t')  
#                if (len(line)!=2):
#                    print(line)
#                b_other.append([str(line[0]),str(line[1])])
#        
        
    def holdout_users(self,test,n):
#        us,bs = zip(*test)
#        C_us = Counter(us)
#        u_num =np.array([ list(C_us.keys()),list(C_us.values())],dtype=np.int)
        ret = []
        for ui in test:
            if ui[0] >n:
                return ret
            else:
                ret.append(ui)
        return ret
                
                        
        

            























