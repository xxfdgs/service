"""
@Name:  main_double.py
@Auth:  rongxing
@Date:  2024/8/14-下午5:14
@IDE:   PyCharm
@PROJECT_NAME:   $ {PROJECT_NAME}
original version is poly_gps/main_pretrain.py
suitable for direct train: double molecule input
monomer_A + monomer_B + Mw or Mn + ratio = Tg
"""

import datetime
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import shutil
import torch
import logging

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_out_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
# from torch_geometric.graphgym.config import (cfg, dump_cfg,
#                                              set_agg_dir, set_cfg, load_cfg,
#                                              makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
### lrx add def modules
from loader_j import create_loader_j
from loader_5 import create_loader_5
from graphgps.config.config_gps import (set_cfg_gps)
from graphgps.create_model_gps import create_model_gps

from graphgps.predicted_finetuning import predicted_load_pretrained_model_cfg, \
    predicted_init_model_from_pretrained
from graphgps.pretrain_finetuning import pretrain_finetuning_load_pretrained_model_cfg, \
    pretrain_finetuning_init_model_from_pretrained
from graphgps.multi_finetuning import multi_load_pretrained_model_cfg,\
    multi_init_model_from_pretrained
from graphgps.lrx_add.read_json import result_picture
from graphgps.lrx_add.json_picture import result_picture_single
from graphgps.lrx_add.predict_average import main_run
from graphgps.lrx_add.predict_average_single import main_run_single
from graphgps.lrx_add.predict_average_multi import main_run_multi

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def set_prediction_output_dir(cfg, cfg_fname):
    """Set custom main output directory path to cfg.
    Store each invocation under ``runs/`` with a timestamped directory name.

    Args:
        cfg_fname (string): Filename for the yaml format configuration file
    """
    config_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cfg.out_dir = os.path.join('runs', f'{timestamp}_{config_name}')
    cfg.save_path = 'runs' + os.sep


def resolve_prediction_csv(csv_path):
    """Resolve the input CSV for post-prediction result aggregation."""
    if os.path.isabs(csv_path) or os.path.isfile(csv_path):
        return csv_path

    dataset_csv_path = os.path.join('datasets_lrx', 'raw', csv_path)
    if os.path.isfile(dataset_csv_path):
        return dataset_csv_path
    return csv_path


def get_inverse_validation_weights(pretrained_dir, repeat_num):
    validation_scores = []
    for seed in range(repeat_num):
        checkpoint_dir = os.path.join(pretrained_dir, str(seed), 'ckpt')
        checkpoint_epochs = [int(name.split('.')[0]) for name in
                             os.listdir(checkpoint_dir) if name.endswith('.ckpt')]
        if not checkpoint_epochs:
            raise FileNotFoundError(f'No checkpoint found in {checkpoint_dir}')

        checkpoint_epoch = max(checkpoint_epochs)
        stats_path = os.path.join(pretrained_dir, str(seed), 'val', 'stats.json')
        with open(stats_path, encoding='utf-8') as stats_file:
            stats = [json.loads(line) for line in stats_file if line.strip()]

        matching_stats = [item for item in stats
                          if item.get('epoch') == checkpoint_epoch]
        if matching_stats:
            validation_scores.append(matching_stats[-1]['mae_sum'])
        else:
            validation_scores.append(min(item['mae_sum'] for item in stats))

    scores = torch.tensor(validation_scores, dtype=torch.float64)
    weights = (1.0 / scores.clamp_min(1e-12))
    weights = weights / weights.sum()
    return weights.tolist()


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    # set_cfg(cfg)
    set_cfg_gps(cfg)
    cfg.train.ckpt_best = True
    load_cfg(cfg, args)
    set_prediction_output_dir(cfg, args.cfg_file)
    dump_cfg(cfg)
    shutil.copy2(args.cfg_file, os.path.join(cfg.out_dir,
                                              os.path.basename(args.cfg_file)))
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.accelerator == 'cuda' and not torch.cuda.is_available():
            logging.warning('CUDA is unavailable; running prediction on CPU.')
            cfg.accelerator = 'cpu'
            cfg.devices = 0
            cfg.device = 'cpu'
        # Set machine learning pipeline
        if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
                or cfg.train.mode == 'double_multi':
            # loaders, loaders_j = create_loader_j()
            loaders, loaders_2, loaders_3, loaders_4, loaders_5 = create_loader_5()
        else:
            loaders = create_loader()

        loggers = create_logger()

        # model = create_model()
        model = create_model_gps() ### set GPU ('cuda', serial)

        if cfg.pretrained.dir:
            if cfg.train.mode == 'double_predict':
                print('---  individual parameter---')
                model = predicted_init_model_from_pretrained(
                    cfg, model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                    cfg.pretrained.reset_prediction_head, seed
                )
            elif cfg.read_multi == True:
                print('')
                model = multi_init_model_from_pretrained(
                    cfg, model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                    cfg.pretrained.reset_prediction_head, seed
                )
            else:
                if (cfg.out_dir).split('/')[1] == 'zinc-GPS+RWSE_pretrain_mask_direct_double':
                    ## 读取平均后的预训练参数
                    print('---  average parameter---')
                    model = init_model_from_pretrained(
                        model, cfg.pretrained.dir, cfg.ave_pretrained_model, cfg.pretrained.freeze_main,
                        cfg.pretrained.reset_prediction_head
                    )
                elif (cfg.out_dir).split('/')[1] == 'pretrain_mask_finetune_double':
                    #### 读取每个初步微调的模型参数 仅使用结构与Tg数据
                    print('---  individual parameter---')
                    model = pretrain_finetuning_init_model_from_pretrained(cfg,
                        model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                        cfg.pretrained.reset_prediction_head, seed
                    )

        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        ### set gpu device
        if cfg.accelerator == 'cuda' and cfg.devices > 0:
            cfg.gpu_device = []
            cfg.gpu_device.append('cuda')
            cfg.gpu_device.append(cfg.gpu_serial)
            # print('cfg.gpu_device', cfg.gpu_device)

        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            if cfg.dataset.data_mask == False and cfg.train.mode != 'predict' and cfg.train.mode != 'double'\
                    and cfg.train.mode != 'double_predict' and cfg.train.mode != 'double_multi':
                train_dict[cfg.train.mode](loggers, loaders, model, optimizer,scheduler)
            # elif cfg.dataset.data_mask == True:
            #     train_dict[cfg.train.mode](loggers, loaders,loaders_j, model, optimizer, scheduler)
            elif cfg.dataset.data_mask == False and cfg.train.mode == 'double':
                train_dict[cfg.train.mode](loggers, loaders,loaders_2, loaders_3, loaders_4, loaders_5,
                                           model, optimizer, scheduler)
            elif cfg.train.mode == 'double_predict':
                train_dict[cfg.train.mode](loggers, loaders, loaders_2, loaders_3, loaders_4, loaders_5,
                                           model, optimizer, scheduler)
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')

    if cfg.train.mode == 'double':
        read_name = (cfg.out_dir).split('/')[1]  #'direct_train_double'
        read_path = cfg.save_path
        repeat_num = args.repeat
        picture_list = 4  # 4 # 1
        if cfg.property_num ==6:
            result_picture(read_path, read_name, repeat_num, 4, cfg.metric_best)
        elif cfg.property_num ==1:
            result_picture_single(read_path, read_name, repeat_num, 4,cfg.metric_best)

    elif cfg.train.mode == 'double_predict':
        file_name = os.path.basename(cfg.out_dir)
        save_path = os.path.dirname(cfg.out_dir) + os.sep
        input_csv = resolve_prediction_csv(cfg.read_csv)
        repeat_num = args.repeat
        model_weights = get_inverse_validation_weights(cfg.pretrained.dir,
                                                       repeat_num)
        picture_list = 4  # 4 # 1
        if cfg.property_num ==6:
            main_run(file_name, save_path, repeat_num, input_csv)
        elif cfg.property_num == 4 or cfg.property_num == 2:
            main_run_multi(file_name, save_path, repeat_num, input_csv,
                           cfg.property_num, model_weights)
        elif cfg.property_num ==1:
            main_run_single(file_name, save_path, repeat_num, input_csv,
                            cfg.property_serial)
