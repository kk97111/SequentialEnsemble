# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:43:07 2018

@author: Yingpeng_Du
"""

from PFMC import PFMC_main
from ANAM import ANAM_main
from HARNN import HARNN_main
from Caser import Caser_main
from FDSA import FDSA_main
from ACF import ACF_main
from SASRec import SASRec_main

factor = 64
seed = 0                                        
batch_size = 2048
#noise_len = 20

#'Kindle'！
for data in  ['Instant_Video','Amazon_App','Kindle','Clothing','Games','Grocery']:
    PFMC_main(data,factor,seed,batch_size,5)
    SASRec_main(data,factor,seed,batch_size)
    Caser_main(data,factor,seed,batch_size)       
    FDSA_main(data,factor,seed,batch_size)
    ACF_main(data,factor,seed,batch_size)
    HARNN_main(data,factor,seed,batch_size)  
    ANAM_main(data,factor,seed,batch_size)
