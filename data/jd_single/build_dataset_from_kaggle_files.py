import os
import pickle
import pdb
import random
import sys

import neotime
from multiprocessing import Pool

from __init__ import data_root
from tqdm import tqdm
from data.utils import get_db_info, create_datapoints_with_xargs, get_neo4j_db_driver



label_feature_name = 'label'
time_key_name = 'time'

neo4j_name_to_feature_name={
    'INDEX_ID':'main_id',
    #'TARGET':'f_action_Product_GetProductDetail',
    'PAIR_ID':  'pair_id',
    'USER_ID':  'user_id',
    'SKU_ID':   'sku_id',
    'TIME':     'time',
}

for i in range(57):
    neo4j_name_to_feature_name['FZ_%d'%(i)]='fz_%d'%(i)

connect_key = ['user_id','sku_id']
sample_limit = [40,40]




db_name = 'jd_single'
target_dir = os.path.join(data_root.replace('home','data'), db_name, 'preprocessed_datapoints_fz_sample')
os.makedirs(target_dir, exist_ok=False)
db_info = get_db_info(db_name)


origin_data_dict = {}
count=0
with open(os.join(data_root.replace('home','data'), 'raw_data',db_name,'train_embedding_100.csv')) as f:
    for each in tqdm(f):
        each = each.strip('\n')
        count+=1
        if count==1:
            feature_to_idx = {feature_name: idx for idx, feature_name in enumerate(each.split(','))}
        else:
            each=each.split(',')
            origin_data_dict[int(each[5])] = each

count=0
with open(os.join(data_root.replace('home','data'), 'raw_data',db_name,'train_embedding_100.csv')) as f:
    for each in tqdm(f):
        each = each.strip('\n')
        count+=1
        if count==1:
            continue
        else:
            each=each.split(',')
            origin_data_dict[int(each[5])] = each

dp_indexs = list(origin_data_dict.keys())


connect_key_dict_list = []
user_dict ={}
city_dict = {}
print('Graph construct')

for key_idx, key in enumerate(connect_key):
    connect_key_dict_list.append({})
    for index, instance in tqdm(origin_data_dict.items()):
        uid = instance[feature_to_idx[key]]
        if uid in connect_key_dict_list[key_idx]:
            connect_key_dict_list[key_idx][uid].append(index)
        else:
            connect_key_dict_list[key_idx][uid] = [index]
    '''
    cid = instance[feature_to_idx['sku_id']]
    if cid in city_dict:
        city_dict[cid].append(index)
    else:
        city_dict[cid] = [index]
    '''



def dump_datapoint(dp_id):
    #for dp_id in tqdm(dp_indexs):
    features = {}
    for node_type in db_info['node_types_and_features'].keys():
        features[node_type] = {}
        for feature_name in db_info['node_types_and_features'][node_type].keys():
            if not feature_name == 'TARGET':
                features[node_type][feature_name] = []


    instance = origin_data_dict[dp_id]
    time = int(instance[feature_to_idx[time_key_name]])
    related_instance_list = []

    for key_idx, key in enumerate(connect_key):
        related_instance_list.append([])
        key_value = instance[feature_to_idx[key]]

        if len(connect_key_dict_list[key_idx][key_value]) < sample_limit[key_idx]:
            candidates_list = connect_key_dict_list[key_idx][key_value]
        else:
            candidates_list = random.sample(connect_key_dict_list[key_idx][key_value],sample_limit[key_idx])
        for each in candidates_list:
            if int(origin_data_dict[each][feature_to_idx[time_key_name]])<=time and each != dp_id:
                related_instance_list[key_idx].append(each)
               
    
    '''
    uid = instance[feature_to_idx['user_id']]
    cid = instance[feature_to_idx['sku_id']]
    time = int(instance[feature_to_idx['time']])

    related_users = []
    related_city = []
    if len(user_dict[uid])<user_sample_limit:
        candidates_user_list = user_dict[uid]
    else:
        candidates_user_list = random.sample(user_dict[uid],user_sample_limit)
    for each in candidates_user_list:
        if int(origin_data_dict[each][feature_to_idx['time']])<=time and each != dp_id:
            related_users.append(each)


    if len(city_dict[cid])<city_sample_limit:
        candidates_city_list = city_dict[cid]
    else:
        candidates_city_list = random.sample(city_dict[cid],city_sample_limit)
    for each in candidates_city_list:
        if int(origin_data_dict[each][feature_to_idx['time']])<=time and each != dp_id:
            related_city.append(each)
    '''


    all_nodes = [dp_id]
    for each in related_instance_list:
        all_nodes +=each
    all_nodes = set(all_nodes)

    node_id_to_graph_idx = {node: idx for idx, node in enumerate(all_nodes)}
    node_types = [0] * (len(all_nodes))

    
    for node_idx in all_nodes:
        node_type = 'Main_table'
        for feature_name, feature_values in features[node_type].items():
            value = origin_data_dict[node_idx][feature_to_idx[neo4j_name_to_feature_name[feature_name]]]
            feature_values.append(value)
    
            
    edge_list = []
    edge_types = []
    for key_idx, _ in enumerate(connect_key):
        for neighbor in related_instance_list[key_idx]:
            edge_list.append((node_id_to_graph_idx[neighbor], 0))
            edge_types.append(key_idx+1)

    '''
    for rel in related_users:
        edge_list.append((node_id_to_graph_idx[rel], 0))
        edge_types.append(1)
    for rel in related_city:
        edge_list.append((node_id_to_graph_idx[rel], 0))
        edge_types.append(2)
    '''

    label = origin_data_dict[dp_id][feature_to_idx[label_feature_name]]

    if dp_id<db_info['task']['n_test']:
        label=None
    elif label=='1':
        label = True
    elif label=='0':
        label = False
    else:
        raise ValueError
    
    with open(os.path.join(target_dir, str(dp_id)), 'wb') as f:
        dp_tuple = (edge_list, node_types, edge_types, features, label)
        pickle.dump(dp_tuple, f)

pool=Pool(15)
res=pool.map(dump_datapoint,range(len(dp_indexs)))
pool.close()
pool.join()
