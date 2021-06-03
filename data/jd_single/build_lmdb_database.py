import base64
import lmdb
import os
import pickle
import pdb
from tqdm import tqdm

from __init__ import data_root

db_name = 'jd_single'
data_path = os.path.join(data_root.replace('home','data'), db_name, 
            'preprocessed_datapoints_fz_sample')

file_list = list(os.listdir(data_path))

os.makedirs(os.path.join(data_path,'lmdb'))
env = lmdb.open(os.path.join(data_path,'lmdb'), map_size=int(1e9)*300) 

txn = env.begin(write=True)  

for i in tqdm(range(len(file_list))):
    filename = file_list[i]
    data = pickle.load(open(os.path.join(data_path,filename),'rb'))
    data = base64.b64encode(pickle.dumps(data)).decode()
    txn.put(key=filename.encode(),value=data.encode())


txn.commit() 
env.close()


