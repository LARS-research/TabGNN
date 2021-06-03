import os
import pdb
import time
import yaml
from datetime import datetime

from __init__ import project_root
from experiments.utils import run_script_with_kwargs

model_class_name = 'GAT'


def get_kwargs(db_name):
    weight_decay = 0.0
    p_dropout = 0.5
    n_layers = 1
    scalar_enc = 'ScalarRobustScalerEnc'  # ScalarRobustScalerEnc ScalarQuantileOrdinalEnc
    datetime_enc = 'DatetimeScalarEnc'  # DatetimeScalarEnc DatetimeOrdinalEnc
    text_enc = 'TextSummaryScalarEnc'  # TextSummaryScalarEnc TfidfEnc
    one_hot_embeddings = False
    readout = 'gap'
    norm = 'none'

    ######################
    # Basic kwargs
    epochs = 50
    if db_name == 'acquirevaluedshopperschallenge':
        max_nodes_per_graph = 25000
        batch_size = 256
        hidden_dim = 128
    elif db_name == 'homecreditdefaultrisk':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 256
    elif db_name == 'kddcup2014':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 256
    elif db_name == 'jd_data':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 128
    elif db_name == 'abc_bank':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 128
    elif db_name == 'jd_small':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 64
    elif db_name == 'luoji':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 64
    elif db_name == 'luoji_no_split':
        max_nodes_per_graph = False
        batch_size = 16
        hidden_dim = 256
    elif db_name == 'luoji_shuffle':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 32
    elif db_name == 'myhug':
        max_nodes_per_graph = False
        batch_size = 256
        hidden_dim = 256
    elif db_name == 'ttgwm':
        max_nodes_per_graph = False
        batch_size = 512
        hidden_dim = 64
    elif db_name == 'yjp':
        max_nodes_per_graph = False
        batch_size = 256
        hidden_dim = 256
    elif db_name == 'jd_single':
        max_nodes_per_graph = False
        batch_size = 256
        hidden_dim = 256
    kwargs = dict(
        seed=1234,
        debug_network=False,
        encoders=dict(
            CATEGORICAL='CategoricalOrdinalEnc',
            SCALAR=scalar_enc,
            DATETIME=datetime_enc,
            LATLONG='LatLongScalarEnc',
            TEXT=text_enc),
        early_stopping_patience=5,
        early_stopping_metric='auroc', #'auroc',
        max_nodes_per_graph=max_nodes_per_graph,
        train_fraction_to_use=1.0,
        dataset_name=db_name,
        device='cuda',
        find_lr=False,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=10
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
            lr=5e-5,
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
        use_jknet=False, 
        cat_fz_embedding=False,
        use_readout=False,
        init_model_class_name='TabMLP',
        init_model_kwargs=dict(
            layer_sizes=[4.0],

            max_emb_dim=8,
            p_dropout=p_dropout,
            one_hot_embeddings=one_hot_embeddings,
            drop_whole_embeddings=False,
            norm_class_name=norm_class_name,
            norm_class_kwargs=norm_class_kwargs,
            activation_class_name='SELU',
            activation_class_kwargs={}
        ),
        n_heads=4,
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
            


if __name__ == '__main__':
    #for db_name in db_names:
    db_name = "jd_single"
    experiment_slug = datetime.now().strftime('%b%d_%H-%M-%S')
    train_test_split = 'use_full_train'


    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    kwargs['optimizer_kwargs'].update(
        lr = config['lr']
    )
    kwargs['model_kwargs']['init_model_kwargs'].update(
        p_dropout = config['p_dropout']
    )
    kwargs['model_kwargs'].update(
        p_dropout = config['p_dropout'],
        hidden_dim = config['hidden_dim']
    )
    kwargs.update(
        epochs = config['epochs'],
        batch_size = config['batch_size'],
        early_stopping_patience = config['early_stopping_patience'],
        num_workers = config['num_workers']
    )

        
    kwargs = get_kwargs(db_name)
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
                            locale='local_tmux',
                            n_gpu=config['gpu_id'],
                            n_cpu=kwargs['num_workers']) 
        
