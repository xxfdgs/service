"""
@Name:  predict.py
@Auth:  rongxing
@Date:  2023/4/13-上午9:25
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
basis on custom_train.py
function: predict smi and output predicted value
"""

import logging
import time

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
### lrx add
import pandas as pd
from graphgps.lrx_add.compute_loss_l1_CL import compute_loss_l1_ntx
from graphgps.lrx_add.compute_loss_l1_CL_2 import compute_loss_l1_ntx_2
from graphgps.lrx_add.compute_loss_l1_CL_3 import compute_loss_l1_ntx_3
from graphgps.lrx_add.compute_loss_l1_CL_3_5component import compute_loss_l1_ntx_3_5component
from graphgps.lrx_add.compute_loss_multi4 import compute_loss_multi4
from graphgps.lrx_add.compute_loss_multi2 import compute_loss_multi2


def loader_loaderj_list(loader,loaders_2,loaders_3,loaders_4,loaders_5):
    batch_list =[]
    for loader_item in loader:
        batch_list_item = []
        batch_list_item.append(loader_item)
        batch_list.append(batch_list_item)
    for serial,loader_item_j in enumerate(loaders_2):
        batch_list[serial].append(loader_item_j)
    for serial,loader_item_j in enumerate(loaders_3):
        batch_list[serial].append(loader_item_j)
    for serial,loader_item_j in enumerate(loaders_4):
        batch_list[serial].append(loader_item_j)
    for serial,loader_item_j in enumerate(loaders_5):
        batch_list[serial].append(loader_item_j)
    return batch_list

@torch.no_grad()
def eval_epoch_multiple(logger, loader, loaders_2, loaders_3, loaders_4, loaders_5, model, split='val'):
    model.eval()
    time_start = time.time()
    list_load = loader_loaderj_list(loader, loaders_2, loaders_3, loaders_4, loaders_5)

    PROPERTY_NUM = cfg.property_num
    batch_size = cfg.train.batch_size

    prediction_frames = []
    for iter,list_ in enumerate(list_load):
        batch = list_[0]
        batch.split = split
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))

        batch,batch_2,batch_3,batch_4,batch_5 = list_[0],list_[1],list_[2],list_[3],list_[4]
        if split == 'train':
            batch.split = 'train'
            batch_2.split = 'train_2'
            batch_3.split = 'train_3'
            batch_4.split = 'train_4'
            batch_5.split = 'train_5'
        elif split == 'val':
            batch.split = 'val'
            batch_2.split = 'val_2'
            batch_3.split = 'val_3'
            batch_4.split = 'val_4'
            batch_5.split = 'val_5'
        elif split == 'test':
            batch.split = 'test'
            batch_2.split = 'test_2'
            batch_3.split = 'test_3'
            batch_4.split = 'test_4'
            batch_5.split = 'test_5'
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_2.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_3.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_4.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_5.to(torch.device(cfg.accelerator, cfg.gpu_serial))


        pred, label = model(batch, batch_2, batch_3, batch_4, batch_5)
        true = label
        if cfg.property_num == 4:  ### 4 property
            loss, pred_score = compute_loss_multi4(pred, true, batch)
            true_ = true.detach().to('cpu', non_blocking=True)
        elif cfg.property_num == 2:  ### 4 property
            loss, pred_score = compute_loss_multi2(pred, true, batch)
            true_ = true.detach().to('cpu', non_blocking=True)
        extra_stats = {}

        _true = true_.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
        #output result detail: pred true
        true_value = _true.numpy()
        pred_value = _pred.numpy()
        # print('true pred over')
        true_value.astype(np.float64)
        pred_value.astype(np.float64)
        A_csv_data = []
        # cut_ = 95 #(batch.sum.detach().to('cpu', non_blocking=True)).numpy()[0]

        # 还原为 [batch_size, 4]（对应“先batch再性质”的平铺方式）
        true_value_ = true_value.reshape(batch_size, PROPERTY_NUM)
        pred_value_ = pred_value.reshape(batch_size, PROPERTY_NUM)

        if cfg.property_num == 4:
            pred_value_ = pred_value_ * 100
            A_csv_data = zip(true_value_[:, 0], pred_value_[:, 0],
                             true_value_[:, 1], pred_value_[:, 1],
                             true_value_[:, 2], pred_value_[:, 2],
                             true_value_[:, 3], pred_value_[:, 3])
            A_csv_name = ['true_EE_before', 'pred_EE_before',
                          'true_EE_after', 'pred_EE_after',
                          'true_Aero_Efficiency', 'pred_Aero_Efficiency',
                          'true_Recovery_Efficiency', 'pred_Recovery_Efficiency']
        elif cfg.property_num == 2:
            A_csv_data = zip(true_value_[:, 0], pred_value_[:, 0],
                             true_value_[:, 1], pred_value_[:, 1])
            A_csv_name = ['true_Norm_before', 'pred_Norm_before',
                          'true_Norm_after', 'pred_Norm_after']
        A_csv = pd.DataFrame(columns=A_csv_name, data=A_csv_data)
        # Only the final padded batch contains artificial samples.  Trim it
        # before appending so several split loaders can safely be evaluated in
        # one prediction invocation.
        if iter == len(list_load) - 1:
            valid_count = int((batch.sum.detach().to(
                'cpu', non_blocking=True)).numpy()[0])
            A_csv = A_csv.iloc[:valid_count]
        prediction_frames.append(A_csv)

    if prediction_frames:
        csv_name = 'test_true_pred_sum.csv'
        A_csv = pd.concat(prediction_frames, axis=0, ignore_index=True)
        read_csv = pd.read_csv(cfg.run_dir + csv_name, index_col=0)
        read_csv_ = pd.concat([read_csv, A_csv], axis=0, ignore_index=True)
        read_csv_.to_csv(cfg.run_dir + csv_name)


@torch.no_grad()
def eval_epoch_single(logger, loader, loaders_2, loaders_3, loaders_4, loaders_5, model, split='val'):
    model.eval()
    time_start = time.time()
    list_load = loader_loaderj_list(loader, loaders_2, loaders_3, loaders_4, loaders_5)

    for iter, list_ in enumerate(list_load):
        batch = list_[0]
        batch.split = split
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))

        batch, batch_2, batch_3, batch_4, batch_5 = list_[0], list_[1], list_[2], list_[3], list_[4]
        if split == 'train':
            batch.split = 'train'
            batch_2.split = 'train_2'
            batch_3.split = 'train_3'
            batch_4.split = 'train_4'
            batch_5.split = 'train_5'
        elif split == 'val':
            batch.split = 'val'
            batch_2.split = 'val_2'
            batch_3.split = 'val_3'
            batch_4.split = 'val_4'
            batch_5.split = 'val_5'
        elif split == 'test':
            batch.split = 'test'
            batch_2.split = 'test_2'
            batch_3.split = 'test_3'
            batch_4.split = 'test_4'
            batch_5.split = 'test_5'
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_2.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_3.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_4.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_5.to(torch.device(cfg.accelerator, cfg.gpu_serial))

        #### single version
        pred, label, gps_feature, middle_feature, pred_feature = model(batch, batch_2, batch_3, batch_4,
                                                                       batch_5)

        extra_stats = {}
        true = label
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
        #output result detail: pred true
        true_value = _true.numpy()
        pred_value = _pred.numpy()
        # print('true pred over')
        true_value.astype(np.float64)
        pred_value.astype(np.float64)
        A_csv_data = []
        cut_ = (batch.sum.detach().to('cpu', non_blocking=True)).numpy()[0]
        # A_csv_data = zip(true_value[:(45)], pred_value[:(45)])
        A_csv_data = zip(true_value[:(cut_)], pred_value[:(cut_)])
        A_csv_name = ['true value', 'pred value']
        A_csv = pd.DataFrame(columns=A_csv_name, data=A_csv_data)
        csv_name = 'test_true_pred_sum.csv'
        read_csv = pd.read_csv(cfg.run_dir + csv_name, index_col=0)
        # print('len(read_csv) + len(A_csv)', len(read_csv), len(A_csv))
        # read_csv_ = pd.merge(read_csv,A_csv,how='outer')
        read_csv_ = pd.concat([read_csv, A_csv], axis=0)
        read_csv_.reset_index(drop=True, inplace=True)
        read_csv_.to_csv(cfg.run_dir + csv_name)

@torch.no_grad()
def eval_epoch(logger, loader,loaders_2, loaders_3, loaders_4, loaders_5, model, cur_epoch, split='val'):
    model.eval()
    time_start = time.time()
    ####loader loader_j
    list_load = loader_loaderj_list(loader,loaders_2,loaders_3,loaders_4,loaders_5)
    for iter,list_ in enumerate(list_load):
        batch = list_[0]
        batch.split = split
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))

        batch,batch_2,batch_3,batch_4,batch_5 = list_[0],list_[1],list_[2],list_[3],list_[4]
        if split == 'val':
            batch.split = 'val'
            batch_2.split = 'val_2'
            batch_3.split = 'val_3'
            batch_4.split = 'val_4'
            batch_5.split = 'val_5'
        elif split == 'test':
            batch.split = 'test'
            batch_2.split = 'test_2'
            batch_3.split = 'test_3'
            batch_4.split = 'test_4'
            batch_5.split = 'test_5'
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_2.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_3.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_4.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_5.to(torch.device(cfg.accelerator, cfg.gpu_serial))

        extra_stats = {}


        if cfg.property_num == 4 or cfg.property_num == 2:
            pred, label = model(batch, batch_2, batch_3, batch_4, batch_5)
        else:
            pred, label, gps_feature, middle_feature, pred_feature = model(batch, batch_2,batch_3,batch_4,batch_5)
        true = label

        if cfg.property_num == 1: ### single property
            loss, pred_score = compute_loss_l1_ntx_3(pred, true, gps_feature, middle_feature, pred_feature,
                                                     batch, cur_epoch)
            true_ = true.detach().to('cpu', non_blocking=True)
        elif cfg.property_num == 6: ### multiply property
            loss,loss_list, pred_score,true_ = compute_loss_l1_ntx_3_5component(pred, true, gps_feature, middle_feature, pred_feature,
                                                     batch, cur_epoch)
            true_ = true_.detach().to('cpu', non_blocking=True)
        elif cfg.property_num == 4: ### 4 property
            loss, pred_score = compute_loss_multi4(pred, true, batch)
            true_ = true.detach().to('cpu', non_blocking=True)
        elif cfg.property_num == 2: ### 4 property
            loss, pred_score = compute_loss_multi2(pred, true, batch)
            true_ = true.detach().to('cpu', non_blocking=True)

        pred_score = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=true_,
                            pred=pred_score,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()



@register_train('double_predict')
def custom_train(loggers, loaders, loaders_2, loaders_3, loaders_4, loaders_5, model, optimizer, scheduler):

    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        print()

    # if cfg.result_out ==False:
    num_splits = len(loggers)
    if cfg.predict_all_splits:
        split_indices = range(num_splits)
        split_names = ['train', 'val', 'test']
    else:
        split_indices = range(1, num_splits - 1)
        split_names = ['test']
    # perf = [[] for _ in range(num_splits)]
    #add csv_sum
    if cfg.property_num == 1:  ### single property
        csv_sum = pd.DataFrame(columns=('true value', 'pred value'))
    elif cfg.property_num == 2:  ### multiply property
        csv_sum = pd.DataFrame(columns=('true_Norm_before', 'pred_Norm_before',
                          'true_Norm_after','pred_Norm_after'))
    elif cfg.property_num == 4:  ### multiply property
        csv_sum = pd.DataFrame(columns=('true_EE_before', 'pred_EE_before',
                          'true_EE_after', 'pred_EE_after',
                          'true_Aero_Efficiency', 'pred_Aero_Efficiency',
                          'true_Recovery_Efficiency','pred_Recovery_Efficiency'))

    csv_sum.to_csv(cfg.run_dir + 'test_true_pred_sum.csv')
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        if is_eval_epoch(cur_epoch):
            for i in split_indices:
                split_name = split_names[i] if cfg.predict_all_splits else split_names[i - 1]
                if cfg.property_num == 1:
                    eval_epoch_single(loggers[i], loaders[i], loaders_2[i], loaders_3[i], loaders_4[i], loaders_5[i]
                               , model, split=split_name)
                elif cfg.property_num == 2 or cfg.property_num == 4:
                    eval_epoch_multiple(loggers[i], loaders[i], loaders_2[i], loaders_3[i], loaders_4[i], loaders_5[i]
                                      , model, split=split_name)

