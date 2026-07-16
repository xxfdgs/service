"""
@Name:  mask_train.py
@Auth:  rongxing
@Date:  2023/8/9-上午10:56
@IDE:   PyCharm
@PROJECT_NAME:   $ {PROJECT_NAME}
basis on custom_train.py
alter it to be suitable for mask pretrain
promote effective GPU
"""

import logging
import time

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

# from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
### lrx add
import pandas as pd
import torch.nn.functional as F

from graphgps.lrx_add.compute_loss_l1_CL import compute_loss_l1_ntx
from graphgps.lrx_add.compute_loss_l1_CL_2 import compute_loss_l1_ntx_2
from graphgps.lrx_add.compute_loss_l1_CL_3 import compute_loss_l1_ntx_3
from graphgps.lrx_add.compute_loss_l1_CL_3_5component import compute_loss_l1_ntx_3_5component
from graphgps.lrx_add.compute_loss_multi4 import compute_loss_multi4
from graphgps.lrx_add.compute_loss_multi2 import compute_loss_multi2
from graphgps.determinism import save_checkpoint_with_metadata


from torch.optim.swa_utils import AveragedModel, SWALR

def batch_j_equal(iter,loader_j):
    for iter_j, batch_j in enumerate(loader_j):
        if iter_j == iter:
            return batch_j
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


def train_epoch(logger, loader,loaders_2, loaders_3, loaders_4, loaders_5,
                model, optimizer, scheduler, cur_epoch, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()

    ####loader loader_5
    list_load = loader_loaderj_list(loader,loaders_2,loaders_3,loaders_4,loaders_5)
    for iter,list_ in enumerate(list_load):
        # print('iter',iter)
        batch,batch_2,batch_3,batch_4,batch_5 = list_[0],list_[1],list_[2],list_[3],list_[4]
        batch.split = 'train'
        batch_2.split = 'train_2'
        batch_3.split = 'train_3'
        batch_4.split = 'train_4'
        batch_5.split = 'train_5'
        batch.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_2.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_3.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_4.to(torch.device(cfg.accelerator, cfg.gpu_serial))
        batch_5.to(torch.device(cfg.accelerator, cfg.gpu_serial))


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

        # true_ = true.detach().to('cpu', non_blocking=True)
        # true_ = true_.detach().to('cpu', non_blocking=True)
        pred_score = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        # print('pred_score= ',pred_score)
        logger.update_stats(true=true_,
                            pred=pred_score,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()



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


@register_train('double')
def custom_train(loggers, loaders,loaders_2, loaders_3, loaders_4, loaders_5, model, optimizer, scheduler):
    """
    Customized training pipeline.
    lrx alter
    add loaders_j
    Args:
        loggers: List of loggers
        loaders: List of loaders
        loaders_j: List of loaders_j
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    patience_count = 0
    best_early_stop_metric = None
    swa_scheduler = SWALR(optimizer, swa_lr=0.0005)
    swa_model = AveragedModel(model)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0],loaders_2[0], loaders_3[0], loaders_4[0], loaders_5[0],
                    model, optimizer, scheduler,cur_epoch,cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i],loaders_2[i], loaders_3[i], loaders_4[i], loaders_5[i]
                , model,cur_epoch, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # if cur_epoch >= 150 and (cur_epoch % 5 ==0):
        #     #### and
        #     logging.info('cur_epoch >= 150')
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()
            # print('swa_scheduler', swa_scheduler.get_last_lr()[0])

        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_checkpoint_with_metadata(model, optimizer, scheduler, cur_epoch)


        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                current_metric = perf[1][-1][m]
                if best_early_stop_metric is None:
                    best_early_stop_metric = current_metric
                    patience_count = 0
                elif cfg.metric_agg == 'argmin':
                    if current_metric < (best_early_stop_metric -
                                         cfg.train.early_stop_min_delta):
                        best_early_stop_metric = current_metric
                        patience_count = 0
                    else:
                        patience_count += 1
                else:
                    if current_metric > (best_early_stop_metric +
                                         cfg.train.early_stop_min_delta):
                        best_early_stop_metric = current_metric
                        patience_count = 0
                    else:
                        patience_count += 1

            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_checkpoint_with_metadata(
                    model, optimizer, scheduler, cur_epoch, perf[1][best_epoch]['loss']
                )
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
        if patience_count >= cfg.train.early_stop_patience:
            # eval_epoch(loggers[i], loaders[i], loaders_2[i], loaders_3[i], loaders_4[i], loaders_5[i]
            #            , model, cur_epoch, split=split_names[i - 1])
            # loggers[i].out_predict()

            print(f'patience_count >= {cfg.train.early_stop_patience}')
            break
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in %s', cfg.run_dir)

