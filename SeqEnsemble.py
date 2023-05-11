from GCNdata import Data
import toolz
import numpy as np
import tensorflow as tf
from time import time
import argparse
import copy
from tqdm import tqdm 
import scipy.sparse as sp
from utils import *
N = 100 #这个N是为了取前多少个items的score，大于N的设为0.
#basemodel
base_model = ['ACF','FDSA','HARNN','Caser','PFMC','SASRec','ANAM']
#'ACF','FDSA','HARNN'都需要attribute信息，Caser','PFMC','SASRec'仅依赖于序列信息。
print_train = False#是否输出train上的验证结果（过拟合解释）。
def parse_args(name,factor,batch_size,tradeoff,user_module,model_module,div_module,epoch,maxlen):
    parser = argparse.ArgumentParser(description="Run .")  
    parser.add_argument('--name', nargs='?', default= name )    
    parser.add_argument('--model', nargs='?', default='SASEM')
    parser.add_argument('--path', nargs='?', default='./datasets/processed/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 0.00001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=epoch)
    parser.add_argument('--tradeoff', type=float, default=tradeoff)
    parser.add_argument('--user_module', nargs='?', default=user_module)
    parser.add_argument('--model_module', nargs='?', default=model_module)    
    parser.add_argument('--div_module', nargs='?', default=div_module)    
    parser.add_argument('--maxlen', type=int, default=maxlen)
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    return parser.parse_args()

class Model(object):
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
        self.n_k = len(base_model)
        self.n_p = self.args.maxlen
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
            
            self.weights = self._initialize_weights()
            self.u = tf.placeholder(tf.int32, shape=[None])
            self.i_pos = tf.placeholder(tf.int32, shape=[None])  
            self.i_neg = tf.placeholder(tf.int32, shape=[None,None])  
            self.input_seq = tf.placeholder(tf.int32, shape=[None,self.n_p])  
            self.meta_pos = tf.placeholder(tf.float32, shape=[None,self.n_k])
            self.meta_neg = tf.placeholder(tf.float32, shape=[None,None,self.n_k])
            self.base_focus = tf.placeholder(tf.int32, shape=[None,self.n_k,self.n_p])
            self.times = tf.placeholder(tf.float32, shape=[None])
            self.is_training = tf.placeholder(tf.bool, shape=())            
            
            #pair interaction   
            self.user_embed = tf.nn.embedding_lookup(self.weights['user_embeddings'],self.u) #[none,d]
            self.condition = tf.to_float(tf.greater(tf.reduce_sum(self.meta_pos,axis=1),0))
            # this is representation of base models (k) that consist top-p items (p) as their representations.
                  
            if self.args.user_module =='MC':   
                self.item_sequence_embs = tf.nn.embedding_lookup(self.weights['item_embeddings1'],self.input_seq)#[none,p,d]      
                self.preference =   self.user_embed + tf.reduce_sum(self.item_sequence_embs,axis=1)
                self.items_embs = self.weights['item_embeddings1']
            if self.args.user_module =='GRU':  
                self.item_sequence_embs = tf.nn.embedding_lookup(self.weights['item_embeddings1'],self.input_seq)#[none,p,d]      
                lstmCell = tf.contrib.rnn.GRUCell(self.hidden_factor)
                value, preference = tf.nn.dynamic_rnn(lstmCell, self.item_sequence_embs, dtype=tf.float32)#[none,5,d]
                self.preference = value[:,-1,:] + self.user_embed #[none,d]
                self.items_embs = self.weights['item_embeddings1']
            if self.args.user_module =='LSTM':  
                self.item_sequence_embs = tf.nn.embedding_lookup(self.weights['item_embeddings1'],self.input_seq)#[none,p,d]      
                lstmCell = tf.contrib.rnn.LSTMCell(self.hidden_factor)
                value, preference = tf.nn.dynamic_rnn(lstmCell, self.item_sequence_embs, dtype=tf.float32)#[none,5,d]
                self.preference = value[:,-1,:] + self.user_embed #[none,d]
                self.items_embs = self.weights['item_embeddings1']
            if self.args.user_module =='SAtt':
                self.state = self.FFN(self.input_seq)
                self.preference =self.state[:,-1,:]  +  self.user_embed #[none,d]
                self.items_embs = self.item_emb_table
            if self.args.user_module =='static':
                self.preference =   self.user_embed 
                self.items_embs = self.weights['item_embeddings1']
                
            if self.args.model_module=='dynamic':
                self.each_model_emb = tf.nn.embedding_lookup(self.items_embs,self.base_focus)#[none,k,p,d]   
                self.wgt_model = tf.reshape(tf.constant(1/np.log2(np.arange(self.n_p)+2),dtype=tf.float32),[1,1,-1,1])
                self.basemodel_emb = self.weights['base_embeddings'] + tf.reduce_sum(self.wgt_model * self.each_model_emb,axis=2)##[none,k,d]self.weights['base_embeddings']
            if self.args.model_module=='static':       
                self.basemodel_emb = self.weights['base_embeddings']#[1,k,d]
            #wgts learning
            self.wgts_org = tf.reduce_sum(tf.expand_dims(self.preference,axis=1)*self.basemodel_emb,axis=-1) #[none,n_k]
            self.wgts = tf.nn.softmax(self.wgts_org,axis=-1)#[none,n_k]            
            self.score_positive = tf.reduce_sum(self.meta_pos * self.wgts,axis=1)#none
            self.score_negative = tf.reduce_sum(self.meta_neg * tf.expand_dims(self.wgts,axis=1),axis=-1)#none * NG
            self.loss_rec = self.pairwise_loss(self.score_positive,self.score_negative)

                                
                
            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)      
            
            
            if self.args.div_module == 'AEM-cov':
                #AEM diversity
                self.model_emb =  self.basemodel_emb#[none,k,p]
                cov_idx = tf.constant(1-np.expand_dims(np.diag(np.ones(self.n_k)),axis=0),dtype=tf.float32)#[none,k,k]    
                cov_div1 =  tf.square(tf.reduce_sum(tf.expand_dims(self.model_emb,axis=1)*tf.expand_dims(self.model_emb,axis=2),axis=-1))
                l2 = tf.reduce_sum(self.model_emb **2,axis=-1)#none* k 
                cov_div2 = tf.matmul(tf.expand_dims(l2,axis=-1),tf.expand_dims(l2,axis=1))#none* k *k
                
                
                
                self.cov =cov_div1/ cov_div2#[none,k,k]           
            if self.args.div_module == 'cov':
                self.model_emb =  self.basemodel_emb#[none,k,p]
                cov_wgt =  tf.stop_gradient(tf.expand_dims(self.wgts,axis=1) + tf.expand_dims(self.wgts,axis=2)) #[none,k,k]
                cov_idx = tf.constant(1-np.expand_dims(np.diag(np.ones(self.n_k)),axis=0),dtype=tf.float32)#[none,k,k]    
                cov_div =  tf.square(tf.reduce_sum(tf.expand_dims(self.model_emb,axis=1)*tf.expand_dims(self.model_emb,axis=2),axis=-1))
                coff = cov_wgt * tf.reshape(self.times,[-1,1,1])
                self.cov =  cov_idx * (1 - cov_div)#[none,k,k]           
                self.cov =  self.cov * coff               
                
            self.loss_diversity = - self.args.tradeoff *  tf.reduce_sum(self.cov)
            
            
            self.loss = self.loss_rec + self.loss_reg + self.loss_diversity
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.meta_all_items = tf.placeholder(tf.float32, shape=[None,self.n_k,self.n_item])            
            out = tf.reduce_sum(tf.expand_dims(self.wgts,axis=2) * self.meta_all_items,axis=1)#[none, n_item]
            self.out_all_topk = tf.nn.top_k(out,200)

            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def FFN(self,input_seq):
        mask = tf.expand_dims(tf.to_float(tf.not_equal(input_seq, -1)), -1)
        reuse = tf.AUTO_REUSE
        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(input_seq,
                                                 vocab_size=self.n_item + 1, #error?
                                                 num_units=self.hidden_factor,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=self.lamda_bilinear,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.item_emb_table = item_emb_table[1:]

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=5,
                num_units=self.hidden_factor,
                zero_pad=False,
                scale=False,
                l2_reg=self.lamda_bilinear,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=0.5,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(2):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_factor,
                                                   num_heads=2,
                                                   dropout_rate=0.5,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_factor,self.hidden_factor],
                                           dropout_rate=0.5, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
        seq_emb = tf.reshape(self.seq, [tf.shape(input_seq)[0] , self.n_p, self.hidden_factor])
        return seq_emb
    def Hessian(self,x):
        return tf.abs(tf.nn.sigmoid(x) * (1- tf.nn.sigmoid(x)) * (1- 2 * tf.nn.sigmoid(x)))

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return tf.Session(config=config)
    
    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['user_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_user, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings1'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings2'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['base_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[1,self.n_k, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['wgts'] =  tf.nn.softmax(tf.Variable(np.random.normal(0.0, 0.01,[1,1,self.n_p,1]),dtype = tf.float32) ,axis=2)# features_M * K
        all_weights['base_embeddings2'] =  tf.Variable(np.random.normal(0.0, 0.01,[1,self.n_k, self.hidden_factor]),dtype = tf.float32) # features_M * K

        return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.u:data['u'],self.input_seq:data['seq'],self.i_pos:data['i_pos'],self.i_neg:data['i_neg'],self.times:data['times'],
                     self.meta_pos:data['meta_pos'],self.meta_neg:data['meta_neg'],self.base_focus:data['base_focus'],self.is_training:True}
        loss_rec,loss_diversity, opt = self.sess.run((self.loss_rec,self.loss_diversity, self.optimizer), feed_dict=feed_dict)
        return loss_rec,loss_diversity

    def pairwise_loss(self,postive,negatvie):
        return -tf.reduce_sum(tf.sigmoid((postive-negatvie)))
    
    def topk(self,user,seq,items_score,base_focus):
        feed_dict = {self.u:user[:,0],self.input_seq:seq,self.meta_all_items:items_score,self.base_focus:base_focus,self.is_training:False}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)     
        wgts = self.sess.run(self.wgts,feed_dict)     
        return self.prediction,wgts
    

class MetaData(object):
    def __init__(self,args,data):
        self.args = args
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.pair2idx = {(line[0],line[1]):i for i,line in enumerate(data.dict_list['user_item'])}
        self.meta = []
        for method in base_model:    
            load = np.load("./datasets/basemodel/%s/%s.npy"%(self.args.name,method))
            self.meta.append(load)
        meta = np.stack(self.meta,axis=1) #this 
        # meta_XXXX denotes [user,item,ranking(500)]
        self.train_meta = meta[[self.pair2idx[line[0],line[1]] for line in self.data.valid]]
        self.test_meta = meta[[self.pair2idx[line[0],line[1]] for line in self.data.test]]
        # 返回对于ensemble的训练集和basemodel值
        self.UI_positive,self.label_data = self.label_positive()

    def all_score(self,traintest):
        #返回所有得分的函数
        rank_chunk = traintest[:,:,2:2+N] #[batch,k,rank]       
        btch,k,n = rank_chunk.shape #[batch,k,rank]
        rank_chunk_reshape = np.reshape(rank_chunk,[-1,n])
        u_k_i = np.zeros([btch*k,self.n_item])     #[batch,k,n_item]
        for i in range(n):
            u_k_i[np.arange(len(u_k_i)),rank_chunk_reshape[:,i]] = 1/(i+10) 
        return np.reshape(u_k_i,[btch,k,self.n_item])     
    def label_positive(self):
        #返回正样本得分的函数        
        n_k = len(base_model)
        label = np.zeros([len(self.train_meta),n_k])
        GT_item = np.expand_dims(self.train_meta[:,0,1],axis=1)#[batch,1] #GT items
        rank_chunk = copy.deepcopy(self.train_meta[:,:,2:2+N]) #[batch,k,rank]       
        for k in range(n_k):
            rank_chunk_k = rank_chunk[:,k,:]    
            torf = GT_item == rank_chunk_k
            label[np.sum(torf,axis=1)>0,k] = 1 / (10 + np.argwhere(torf)[:,1])
        return self.train_meta[:,0,:2],label
    def label_negative(self,neglist,NG):#neglist is 1-d list where each element denotes the negative item
        #返回负样本得分的函数        
        n_k = len(base_model)
        assert len(neglist)==len(self.train_meta), 'wrong size'
        label = []
        for i in range(NG):
            label_i = np.zeros([len(self.train_meta),n_k])
            GT_item = np.expand_dims(neglist[:,i],axis=1)#[batch,1] #for item
            rank_chunk = copy.deepcopy(self.train_meta[:,:,2:2+N]) #[batch,k,rank]       
            for k in range(n_k):
                rank_chunk_k = rank_chunk[:,k,:]
                torf = GT_item == rank_chunk_k
                label_i[np.sum(torf,axis=1)>0,k] = 1 / (10+ np.argwhere(torf)[:,1])
            label.append(label_i)
        return np.stack(label,axis=1)
        
            
class Train_MF(object):
    def __init__(self,args,data,meta_data):
        self.args = args
        self.epoch = self.args.epoch
        self.batch_size = args.batch_size
        self.n_p = args.maxlen        
        # Data loadin
        self.data = data
        self.meta_data = meta_data
        self.entity = self.data.entity
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
    # Training\\\建立模型
        self.model = Model(self.args,self.data ,args.hidden_factor,args.lr, args.lamda, args.optimizer)
        with open("./process_result.txt","a") as f:
             f.write("dataset:%s\n"%(args.name))

    def train(self):  # fit a dataset
        #初始结果   
        MAP_valid = 0
        p=0
        for epoch in range(0,self.epoch+1): #每一次迭代训练
            #shuffle
            shuffle = np.arange(len(self.meta_data.UI_positive))
#            np.random.shuffle(shuffle)
            #sample负样本采样
            ui = self.meta_data.UI_positive #none * 2
            self.u = ui[:,0]   
            self.t = self.timestamp()
#            positive samples
            self.i_pos = ui[:,1]
            
            NG = 1
            self.i_neg = self.sample_negative(ui,self.meta_data.train_meta,NG)#采样，none * NG
            self.seq =np.array([self.data.latest_interaction[(line[0],line[1])] for line in ui])#none*seq        
            meta_positive = self.meta_data.label_data #none * k  k denotes BM number
            meta_negative = self.meta_data.label_negative(self.i_neg,NG)#none * NG * k
            base_focus = self.meta_data.train_meta[:,:,2:2+self.n_p] #none * k * p # p denotes window size
            
            for user_chunk in tqdm(toolz.partition_all(self.batch_size,[i for i in range(len(ui))] )):                
                p = p + 1
                chunk = shuffle[list(user_chunk)]
                u_chunk = self.u[chunk] #none 
                seq_chunk = self.seq[chunk]#none * p
                i_pos_chunk = self.i_pos[chunk] #none 
                i_neg_chunk = self.i_neg[chunk] #none * NG
                meta_positive_chunk = meta_positive[chunk] #none *k
                meta_negative_chunck = meta_negative[chunk]#none * NG *k
                base_focus_chunck = base_focus[chunk]
                times = self.t[chunk]
                self.feed_dict = {'u':u_chunk,'seq':seq_chunk,'i_pos':i_pos_chunk,'i_neg':i_neg_chunk,
                                  'meta_pos':meta_positive_chunk,'meta_neg':meta_negative_chunck,
                                  'base_focus':base_focus_chunck,'times':times}
                loss =  self.model.partial_fit(self.feed_dict)
         # evaluate training and validation datasets
            if epoch % 1 ==0:
                print("Loss %.4f\t%.4f"%(loss[0],loss[1]))
                if print_train:
                    init_test_TopK_train = self.evaluate_TopK(self.data.valid[:10000],self.meta_data.train_score[:10000],self.meta_data.train_meta[:10000],[10]) 
                    print(init_test_TopK_train)
                init_test_TopK_test = self.evaluate_TopK(self.data.test,0,self.meta_data.test_meta,[20,50]) #0 = self.meta_data.test_score
                init_test_TopK_valid =0,0,0
                print("Epoch %d \t TEST SET:%.4f MAP:%.4f,NDCG:%.4f,PREC:%.4f\n"
                  %(epoch,init_test_TopK_valid[2],init_test_TopK_test[3],init_test_TopK_test[4],init_test_TopK_test[5]))
                with open("./process_result.txt","a") as f:
                     f.write("Epoch %d \t TEST SET:%.4f MAP:%.4f,NDCG:%.4f,PREC:%.4f\n"
                      %(epoch,init_test_TopK_valid[2],init_test_TopK_test[3],init_test_TopK_test[4],init_test_TopK_test[5]))

                if MAP_valid < np.mean(init_test_TopK_test[4:]):
                    MAP_valid = np.mean(init_test_TopK_test[4:])
                    result_print = init_test_TopK_test
        with open("./result.txt","a") as f:
            f.write("%s,%s,%s,%s,%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n"%(self.args.name,self.args.model,self.args.user_module,self.args.model_module,self.args.div_module,self.args.tradeoff,result_print[0],result_print[1],result_print[2],result_print[3],result_print[4],result_print[5]))
    def sample_negative(self, u_i,meta_data,NG):  
        BM_train = self.data.dict_forward['train']
        M = 50
        meta_data = meta_data[:,:,2:2+M]
        meta_data = np.reshape(meta_data,[len(meta_data),-1]) #[none,k*500]
        na,nb = meta_data.shape
        sample = []
        for i in range(NG):            
            sample_i = np.random.randint(0,self.n_item,na)
            for j,item in enumerate(sample_i):
                if item in BM_train[u_i[j,0]]:
                    sample_i[j] =  np.random.randint(0,self.n_item) 
            sample.append(sample_i)
        
        return np.stack(sample,axis=-1)
    def timestamp(self):
        # infer timestamps
        t = np.ones(len(self.u))
        s,cout,c = self.u[0],10,1
        for i,ur in enumerate(self.u):
            if ur==s:
                cout+=1  
                c+=1
                t[i] =cout                                          
            else:
                t[i-c:i] = t[i-c:i]/cout
                cout=10
                c = 1
                s = ur
                t[i] =cout 
        t[i-c+1:] = t[i-c+1:]/cout
        return t

    def evaluate_TopK(self,test,test_score,test_meta,topk):
        u_i = copy.deepcopy(np.array(test))#none * 2
        size = len(u_i)
        result_MAP = {key:[] for key in topk}
        result_PREC = {key:[] for key in topk}
        result_NDCG = {key:[] for key in topk}
        num = 999#self.n_user
        last_iteraction = [] #none*5
        for line in u_i:
            user,item = line
            last_iteraction.append(self.data.latest_interaction[(user,item)])
        last_iteraction = np.array(last_iteraction)
        for _ in range(int(size/num+1)):
            beg,end = _*num,(_+1)*num
            ui_block = u_i[beg:end]
            last_iteraction_block = last_iteraction[beg:end]
#            items_score = test_score[beg:end]
            items_score = self.meta_data.all_score(test_meta[beg:end])
            
            base_focus = test_meta[beg:end,:,2:2+self.n_p]
            self.score,self.wgts = self.model.topk(ui_block,last_iteraction_block,items_score,base_focus) #none * 50
            prediction = self.score 
            assert len(prediction) == len(ui_block)
            for i,line in enumerate(ui_block):
                for key in topk:
                    user,item = line
                    n = 0 
                    for it in prediction[i]:
                        if n> key -1:
                            result_MAP[key].append(0.0)
                            result_NDCG[key].append(0.0)
                            result_PREC[key].append(0.0)  
                            n=0
                            break
                        elif it == item:   
                            result_MAP[key].append(1.0)
                            result_NDCG[key].append(np.log(2)/np.log(n+2))
                            result_PREC[key].append(1/(n+1))
                            n=0
                            break
                        elif it in self.data.set_forward['train'][user] or it in self.data.set_forward['valid'][user]:
                            continue
                        else:
                            n = n + 1   
        return  [np.mean(result_MAP[topk[0]]),np.mean(result_NDCG[topk[0]]),np.mean(result_PREC[topk[0]]),np.mean(result_MAP[topk[1]]),np.mean(result_NDCG[topk[1]]),np.mean(result_PREC[topk[1]])] 

  
def SEM_main(name,factor,batch_size,tradeoff,user_module,model_module,div_module,epoch,maxlen):

    args = parse_args(name,factor,batch_size,tradeoff,user_module,model_module,div_module,epoch,maxlen)
    data = Data(args,0)#获取数据
    meta_data = MetaData(args,data)
    session_DHRec = Train_MF(args,data,meta_data)
    session_DHRec.train()