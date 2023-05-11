
#from diversity_evoluation import diversity
from SeqEnsemble import SEM_main
factor = 32
seed = 0          
# Gems 2048                               
batch_size = {'Amazon_App':1024,'Kindle':1024,'Clothing':2048,'Grocery':2048,'Instant_Video':256,'Games':1024}
tradeoff = {'Amazon_App':1,'Clothing':2,'Grocery':128,'Kindle':128,'Instant_Video':2,'Games':32}
epoch = {'Amazon_App':15,'Kindle':5,'Clothing':10,'Grocery':20,'Instant_Video':20,'Games':10}
maxlen = {'Amazon_App':5,'Kindle':3,'Clothing':3,'Grocery':5,'Instant_Video':5,'Games':5}

#Setting for the proposed method and the ablations.
#method_name:tradeoff,user_module,model_module,div_module.
#SEM:tradeoff[data],'SAtt','dynamic','cov'.
#w/o uDC:tradeoff[data],'static','dynamic','cov'.
#w/o bDE:tradeoff[data],'SAtt','static','cov'.
#w/o Div:0.0,'SAtt','dynamic','cov'.
#w/o TPDiv:tradeoff[data],'SAtt','dynamic','AEM-cov'.

#example:
data = 'Kindle'
SEM_main(data,factor,batch_size[data],tradeoff[data],'SAtt','dynamic','cov',epoch[data],maxlen[data])           
