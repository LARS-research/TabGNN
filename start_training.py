import base64
import json
import logging
import os
import pdb
import pickle
import pprint
import setproctitle
import sys
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as opt
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, log_loss, accuracy_score
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, ExponentialLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

import models
from data.DatabaseDataset import DatabaseDataset
from data.TabularDataset import TabularDataset
from data.utils import write_kaggle_submission_file
from models.GNN.GNNModelBase import GNNModelBase
from models.tabular.TabModelBase import TabModelBase
from models.utils import save_train_kwargs, recursive_to, save_model_checkpoint, get_good_lr, register_module_hooks
from utils import setup_writer, get_train_val_test_datasets, get_dataloader, log_param_values, \
    format_hparam_dict_for_tb, model_to_device, get_optim_with_correct_wd


def train_epoch(writer, train_loader, model, optimizer, scheduler, epoch):
    model.train()
    writer.batches_done = epoch * len(train_loader)
    # t = time.perf_counter()
    loss_sum = []
    for batch_idx, (input, label) in enumerate(tqdm(train_loader)):
        recursive_to((input, label), model.device)
        input = list(input)
        input[0] = input[0].to(model.device)
        input = tuple(input)
        optimizer.zero_grad()
        output = model(input)
        loss = model.loss_fxn(output, label)
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')
        loss.backward()
        loss_sum.append(float(loss))
        optimizer.step()
        writer.batches_done += 1
        log_param_values(writer, model)
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()
    if isinstance(scheduler, (ExponentialLR, CosineAnnealingWarmRestarts)):
        scheduler.step()
    
    return sum(loss_sum)/len(loss_sum)


def validate_model(writer, val_loader, model, epoch):
    model.eval()
    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        n_correct = torch.Tensor([0])
        labels = []
        probs = []
        for batch_idx, (input, label) in enumerate(tqdm(val_loader)):
            recursive_to((input, label), model.device)
            input = list(input)
            input[0] = input[0].to(model.device)
            input = tuple(input)
            output = model(input)
            val_loss += model.loss_fxn(output, label).cpu()  # sum up mean batch losses
            if isinstance(output, torch.Tensor):
                probs.append(torch.softmax(output, dim=1).cpu())
                pred = model.pred_from_output(output)
                n_correct += pred.eq(label.view_as(pred)).sum().cpu()
                labels.append(label.cpu())

        val_loss = (val_loss.cpu() / len(val_loader)).item()
        #writer.add_scalar('Validation/{}'.format(model.loss_fxn.__class__.__name__), val_loss, writer.batches_done)
        print(f'val_loss epoch {epoch}: {val_loss}')

        if isinstance(output, torch.Tensor):
            labels = torch.cat(labels, dim=0).cpu().numpy()
            probs = torch.cat(probs, dim=0).cpu().numpy()

            val_acc = (100 * n_correct / len(val_loader.dataset)).item()
            print(f'val_acc epoch {epoch}: {val_acc}')

            val_auroc = roc_auc_score(labels, probs[:, 1])
            print(f'val_auroc epoch {epoch}: {val_auroc}')

        else:
            val_auroc = val_acc = None

        return val_auroc, val_acc, val_loss

def train_model(
        writer,
        seed,
        log_dir,
        debug_network,
        dataset_name,
        train_test_split,
        encoders,
        max_nodes_per_graph,
        train_fraction_to_use,
        sampler_class_name,
        sampler_class_kwargs,
        model_class_name,
        model_kwargs,
        batch_size,
        epochs,
        optimizer_class_name,
        optimizer_kwargs,
        lr_scheduler_class_name,
        lr_scheduler_kwargs,
        early_stopping_patience,
        wd_bias,
        wd_embed,
        wd_bn,
        load_model_weights_from='',
        early_stopping_metric='loss',
        device='cpu',
        num_workers=0,
        find_lr=True):
    setproctitle.setproctitle("RDB2Graph_%s@quanyuhan"%(dataset_name))
    train_data, val_data, _ = get_train_val_test_datasets(dataset_name=dataset_name,
                                                          train_test_split=train_test_split,
                                                          encoders=encoders,
                                                          train_fraction_to_use=train_fraction_to_use)

    train_loader = get_dataloader(dataset=train_data,
                                  batch_size=batch_size,
                                  sampler_class_name=sampler_class_name,
                                  sampler_class_kwargs=sampler_class_kwargs,
                                  num_workers=num_workers,
                                  max_nodes_per_graph=max_nodes_per_graph)
    print(f'Batches per train epoch: {len(train_loader)}')
    print(f'Total batches: {len(train_loader) * epochs}')
    val_loader = get_dataloader(dataset=val_data,
                                batch_size=batch_size,
                                sampler_class_name='SequentialSampler',
                                num_workers=num_workers,
                                max_nodes_per_graph=max_nodes_per_graph)

    def init_model():
        model_class = models.__dict__[model_class_name]
        if isinstance(train_data, TabularDataset):
            assert issubclass(model_class, TabModelBase)
            model_kwargs.update(
                n_cont_features=train_data.n_cont_features,
                cat_feat_origin_cards=train_data.cat_feat_origin_cards
            )
        elif isinstance(train_data, DatabaseDataset):
            assert issubclass(model_class, GNNModelBase)
            model_kwargs.update(
                feature_encoders=train_data.feature_encoders
            )
        else:
            raise ValueError
        model = model_class(writer=writer,
                            dataset_name=dataset_name,
                            **model_kwargs)
        if load_model_weights_from:
            state_dict = torch.load(load_model_weights_from, map_location=torch.device('cpu'))
            retval = model.load_state_dict(state_dict['model'], strict=False)
            print(f'Missing modules:\n{pprint.pformat(retval.missing_keys)}')
            print(f'Unexpected modules:\n{pprint.pformat(retval.unexpected_keys)}')
        
        model_to_device(model, device)

        # If debugging, add hooks to all modules
        if debug_network:
            register_module_hooks('model', model, writer)

        return model

    # Optionally find good learning rate
    if find_lr:
        print('Finding good learning rate')
        model = init_model()
        optimizer = get_optim_with_correct_wd(optimizer_class_name, model, optimizer_kwargs, wd_bias, wd_embed, wd_bn)
        good_lr = get_good_lr(model, optimizer, train_loader, init_value=1e-7, final_value=1.0, beta=0.98)
        optimizer_kwargs.update(lr=good_lr)
        writer.train_kwargs['optimizer_kwargs'].update(lr=good_lr)
        if lr_scheduler_class_name == 'CyclicLR':
            lr_scheduler_kwargs.update(max_lr=good_lr, base_lr=good_lr / 100)
            writer.train_kwargs['lr_scheduler_kwargs'].update(max_lr=good_lr, base_lr=good_lr / 100)
        elif lr_scheduler_class_name == 'OneCycleLR':
            lr_scheduler_kwargs.update(max_lr=good_lr)
            writer.train_kwargs['lr_scheduler_kwargs'].update(max_lr=good_lr)

    model = init_model()
    optimizer = get_optim_with_correct_wd(optimizer_class_name, model, optimizer_kwargs, wd_bias, wd_embed, wd_bn)
    scheduler = opt.lr_scheduler.__dict__[lr_scheduler_class_name](optimizer, **lr_scheduler_kwargs)
    
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'runs', log_dir, 'train.log')):
        logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'runs', log_dir, 'train_log'), level=logging.INFO, filemode='w')
        logging.info('start training')
        print('logging path: %s'%(os.path.join(os.path.dirname(__file__), 'runs', log_dir, 'train_log')))
    else:
        print(os.path.join('./runs/', log_dir, 'train.log'), 'already exists, manually delete it!')
        exit()

    
    

    # Run train loop with early stopping
    best_auroc = -1
    best_acc = -1
    best_loss = np.inf
    best_epoch = -1
    try:
        for epoch in range(epochs):
            log_param_values(writer, model)
            if epoch % 20 == 0:
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler)
            
            start = time.time()
            train_loss = train_epoch(writer, train_loader, model, optimizer, scheduler, epoch)
            train_time = time.time()-start

            start = time.time()
            val_auroc, val_acc, val_loss = validate_model(writer, val_loader, model, epoch)
            val_time = time.time()-start

            best = False
            if val_auroc is not None and val_auroc > best_auroc:
                best_auroc = val_auroc
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_auroc')
                if early_stopping_metric == 'auroc':
                    best = True
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_acc')
                if early_stopping_metric == 'acc':
                    best = True
            if train_loss < best_loss:
                best_loss = train_loss
                save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_loss')
                if early_stopping_metric == 'loss':
                    best = True
            if early_stopping_metric == 'auroc':
                m = val_auroc
            elif early_stopping_metric == 'acc':
                m = val_acc
            elif early_stopping_metric == 'loss':
                m = -1 * val_loss
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(m)
            if best:
                best_epoch = epoch
            
            if epoch - best_epoch >= early_stopping_patience:
                Path(os.path.join(writer.log_dir, 'stopped_early.info')).touch()
                break

            logging.info('epoch %d, [%.1fs+%.1fs] || train_loss = %.4f, val_loss = %.4f, val_auc = %.5f'%(epoch, train_time, val_time, train_loss, val_loss, val_auroc))
            print('epoch %d, [%.1fs+%.1fs] || train_loss = %.4f, val_loss = %.4f, val_auc = %.5f'%(epoch, train_time, val_time, train_loss, val_loss, val_auroc))

            if hasattr(model, 'prune'):
                model.prune(epoch, m)
        else:
            save_model_checkpoint(writer, epoch, model, optimizer, scheduler, chkpt_name='best_loss')
            validate_model(writer, val_loader, model, epoch)
            Path(os.path.join(writer.log_dir, 'finished_all_epochs.info')).touch()
        writer.add_hparams(format_hparam_dict_for_tb(writer.train_kwargs), {'hparam/best_auroc': best_auroc,
                                                                            'hparam/best_acc': best_acc,
                                                                            'hparam/best_loss': best_loss,
                                                                            'hparam/best_epoch': best_epoch})
    except Exception as e:
        Path(os.path.join(writer.log_dir, 'failed.info')).touch()
        writer.add_hparams(format_hparam_dict_for_tb(writer.train_kwargs), {'hparam/best_auroc': best_auroc,
                                                                            'hparam/best_acc': best_acc,
                                                                            'hparam/best_loss': best_loss,
                                                                            'hparam/best_epoch': best_epoch})
        raise e

def main(kwargs):
    # Workaround for pytorch large-scale multiprocessing issue, if you're using a lot of dataloaders
    # torch.multiprocessing.set_sharing_strategy('file_system')

    torch.manual_seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])

    writer = setup_writer(kwargs['log_dir'], kwargs['debug_network'])
    save_train_kwargs(writer, kwargs)
    writer.add_text('train_kwargs', pprint.pformat(kwargs).replace('\n', '\t\n'))
    writer.train_kwargs = kwargs

    if kwargs['model_class_name'] == 'LightGBM':
        train_non_deep_model(writer, **kwargs)
    else:
        train_model(writer, **kwargs)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        kwargs = dict()
    else:
        kwargs_filename = sys.argv[1]
        kwargs=pickle.load(open(kwargs_filename, 'rb'))
        #kwargs = pickle.loads(base64.b64decode(sys.argv[1]))
        #os.remove(kwargs_filename)
    main(kwargs)
