import os
import pdb
import time
from datetime import datetime

import hyperopt
from hyperopt import hp, fmin, tpe, space_eval, Trials, partial

from __init__ import project_root
from experiments.utils import run_script_with_kwargs
from start_training import main as main_training

model_class_name = 'ERGAT'

db_names = (
    #'acquirevaluedshopperschallenge',
    #'homecreditdefaultrisk',
    #'kddcup2014',
    'abc_bank',
)


def get_kwargs(db_name, space):
    weight_decay = space['weight_decay']
    p_dropout = space['dropout']
    n_layers = space['layers']
    scalar_enc = 'ScalarRobustScalerEnc'  # ScalarRobustScalerEnc ScalarQuantileOrdinalEnc
    datetime_enc = 'DatetimeScalarEnc'  # DatetimeScalarEnc DatetimeOrdinalEnc
    text_enc = 'TextSummaryScalarEnc'  # TextSummaryScalarEnc TfidfEnc
    one_hot_embeddings = False
    readout = space['readout']
    norm = space['norm']

    ######################
    # Basic kwargs
    epochs = 100
    if db_name == 'acquirevaluedshopperschallenge':
        max_nodes_per_graph = 20000
        batch_size = 32
        hidden_dim = 64
    elif db_name == 'homecreditdefaultrisk':
        max_nodes_per_graph = False
        batch_size = 256
        hidden_dim = 256
    elif db_name == 'kddcup2014':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 256
    elif db_name == 'jd_data':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 256
    elif db_name == 'abc_bank':
        max_nodes_per_graph = False
        batch_size = space['batchsize']
        hidden_dim = space['hidden_dim']
    elif db_name == 'jd_small':
        max_nodes_per_graph = False
        batch_size = space['batchsize']#512
        hidden_dim = space['hidden_dim']#128
    kwargs = dict(
        seed=1234,
        debug_network=False,
        encoders=dict(
            CATEGORICAL='CategoricalOrdinalEnc',
            SCALAR=scalar_enc,
            DATETIME=datetime_enc,
            LATLONG='LatLongScalarEnc',
            TEXT=text_enc),
        early_stopping_patience=10,
        early_stopping_metric='auroc',
        max_nodes_per_graph=max_nodes_per_graph,
        train_fraction_to_use=1.0,
        dataset_name=db_name,
        device='cuda',
        find_lr=False,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=8
    )
    # LR Schedule
    kwargs.update(
        lr_scheduler_class_name='StepLR',
        lr_scheduler_kwargs=dict(
            step_size=1,
            gamma=1.0
        ),
    )
    # Optimizer
    kwargs.update(
        optimizer_class_name='AdamW',
        optimizer_kwargs=dict(
            lr=space['learning_rate'],
            weight_decay=weight_decay,
        ),
        wd_bias=False,
        wd_embed=False,
        wd_bn=False,
    )
    # Sampler
    sampler_class_name = 'RandomSampler'
    sampler_class_kwargs = {}
    kwargs.update(sampler_class_name=sampler_class_name,
                  sampler_class_kwargs=sampler_class_kwargs)

    # Normalization kwargs
    if norm == 'none':
        norm_class_name = 'Identity'
        norm_class_kwargs = dict()
    elif norm == 'batchnorm':
        norm_class_name = 'BatchNorm1d'
        norm_class_kwargs = dict()
    elif norm == 'layernorm':
        norm_class_name = 'LayerNorm'
        norm_class_kwargs = dict()

    # Model
    kwargs.update(
        model_class_name=model_class_name,
        model_kwargs=dict()
    )
    kwargs['model_kwargs'].update(
        hidden_dim=hidden_dim,
        init_model_class_name='TabMLP',
        use_jknet=False, 
        cat_fz_embedding=False,
        init_model_kwargs=dict(
            # n_cont_embeddings=1,
            # n_heads=1,
            # readout='mean',
            # n_layers=8,
            # hidden_dim=512,
            # # attn_temp=1.0,
            # column_embedding=False,
            # orig_emb_resid=False,

            layer_sizes=space['layer_sizes'],

            max_emb_dim=32,
            p_dropout=p_dropout,
            one_hot_embeddings=one_hot_embeddings,
            drop_whole_embeddings=False,
            norm_class_name=norm_class_name,
            norm_class_kwargs=norm_class_kwargs,
            activation_class_name='SELU',
            activation_class_kwargs={}
        ),
        n_heads=space['n_heads'],
        residual=True,

        p_dropout=p_dropout,
        n_layers=n_layers,
        activation_class_name='SELU',
        activation_class_kwargs={},
        norm_class_name=norm_class_name,
        norm_class_kwargs=norm_class_kwargs,
        loss_class_name='CrossEntropyLoss',
        loss_class_kwargs=dict(
            weight=None,
        ),
        fcout_layer_sizes=[],
    )
    ######################
    # Readout kwargs
    if readout == 'avg':
        readout_class_name = 'AvgPooling'
        readout_kwargs = dict()
    elif readout == 'sort':
        readout_class_name = 'SortPooling'
        readout_kwargs = dict(k=5)
    elif readout == 'gap':
        readout_class_name = 'GlobalAttentionPooling'
        readout_kwargs = dict(n_layers=2,
                              act_name='SELU')
    elif readout == 's2s':
        readout_class_name = 'Set2Set'
        readout_kwargs = dict(n_iters=2,
                              n_layers=2)
    elif readout == 'std':
        readout_class_name = 'SetTransformerDecoder'
        readout_kwargs = dict(p_dropout=p_dropout,
                              num_heads=2,
                              n_layers=2,
                              k=3)
    kwargs['model_kwargs'].update(
        readout_class_name=readout_class_name,
        readout_kwargs=readout_kwargs
    )

    return kwargs



def objective(space):
    db_name = space['db_name']
    experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S')
    train_test_split = space['train_test_split']

    kwargs = get_kwargs(db_name, space)
    kwargs['log_dir'] = os.path.join('GNN',
                                        db_name,
                                        model_class_name,
                                        experiment_slug,
                                        train_test_split)
    kwargs['train_test_split'] = train_test_split
    session_name = '_'.join([db_name, model_class_name, experiment_slug, train_test_split])

    run_script_with_kwargs('start_training',
                            kwargs,
                            session_name,
                            locale='no_tmux',
                            n_gpu=6,
                            n_cpu=kwargs['num_workers'],
                            mb_memory=60000)  # this is the memory on a p3.2xlarge
    #main_training(kwargs)
    
    time.sleep(5) # wait for create log dir
    while 1:
        train_path = os.path.join(project_root, 'runs', 'GNN', str(db_name), 'ERGAT', experiment_slug, train_test_split)
        file_list = os.listdir(train_path)
        
        if 'stopped_early.info' in file_list or 'finished_all_epochs.info' in file_list:
            break
        else:
            time.sleep(300)
    
    best_val_auc = 0
    with open(os.path.join(train_path, 'train_log')) as f:
        count=0
        for each in f:
            count+=1
            if count==1:
                continue
            else:
                val_auc = float(each.strip('\n').split(',')[-1].replace(' val_auc = ',''))
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
    
    return -best_val_auc






if __name__ == '__main__':
    #for db_name in db_names:
    db_name = "abc_bank"
    #experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S-%f')
    for train_test_split in [
        'use_full_train',
        #'xval0',
        #'xval1',
        #'xval2',
        #'xval3',
        #'xval4'
    ]:


        space = {
            'readout': 'gap', #hp.choice('readout', ['avg', 'sort', 'gap', 's2s']),
            'norm': hp.choice('norm', ['none', 'batchnorm', 'layernorm']),
            'layer_sizes':  hp.choice('layer_sizes', [[4.0], [3.0], [2.0],[1.0]]),
            'n_heads':  hp.choice('n_heads', [2,4]),
            'dropout': hp.uniform('dropout', 0, 1),
            'weight_decay': hp.loguniform('weight_decay',-10, -3),
            'learning_rate': hp.loguniform("learning_rate", -10, -3),
            'batchsize': hp.choice('batchsize', [128, 256]),
            'hidden_dim': hp.choice('hidden_dim', [64,128]),
            'layers': 1 ,
            'db_name': db_name,
            'train_test_split': train_test_split,
        }

        trials = Trials()    
        best = fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=15), max_evals=100, trials=trials, show_progressbar=False)

        best_space = hyperopt.space_eval(space, best)
        print('best space is ', best_space)



        '''
        kwargs = get_kwargs(db_name)
        kwargs['log_dir'] = os.path.join('ERGNN',
                                            db_name,
                                            model_class_name,
                                            experiment_slug,
                                            train_test_split)
        kwargs['train_test_split'] = train_test_split
        session_name = '_'.join([db_name, model_class_name, experiment_slug, train_test_split])
        run_script_with_kwargs('start_training',
                                kwargs,
                                session_name,
                                locale='AWS_Batch',
                                n_gpu=1,
                                n_cpu=kwargs['num_workers'],
                                mb_memory=60000)  # this is the memory on a p3.2xlarge
        '''