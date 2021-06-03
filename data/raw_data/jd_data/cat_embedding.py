import numpy as np
import pandas as pd
import pdb
import pickle



train=pd.read_csv('./train.csv')
test=pd.read_csv('./test.csv')

#train=train.drop('index_id',axis=1)
#test=test.drop('index_id',axis=1)


embedding=pd.read_csv('./fz_embedding.csv')

embedding_test=embedding.iloc[0:test.shape[0],:]
embedding_train=embedding.iloc[test.shape[0]:,:]


test=pd.merge(test,embedding,on='index_id', how='left')
train=pd.merge(train,embedding,on='index_id', how='left')
#train=pd.concat([train,embedding_train.reset_index(drop=True)],axis=1)
#test=pd.concat([test,embedding_test],axis=1)


train.to_csv('train_embedding.csv',index=False)
test.to_csv('test_embedding.csv',index=False)






