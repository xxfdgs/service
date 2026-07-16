import logging
import os.path as osp
import time
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import (GNNBenchmarkDataset, Planetoid, TUDataset,
                                      WikipediaNetwork, ZINC)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader


from graphgps.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)
import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
# config_path = CURRENT_DIR.rsplit('/', 3)[0]  # 当前目录,可以通过修改分割最右边的第几个'/'来拿到第几层目录
# sys.path.append(config_path + cfg.filename)
sys.path.append(CURRENT_DIR.rsplit('/', 2)[0])

import os, glob
from os import listdir
##### masked subgraph
from mask_subgraph.csv_pyg_mask import LRX_mask_subgraph
from mask_subgraph.csv_pyg_classify import LRX_mask_classify
from mask_subgraph.csv_pyg_mask_one import LRX_mask_subgraph_one

from graphgps.lrx_add.csv_pyg_five import LRX_five
from graphgps.lrx_add.csv_pyg_five_predict import LRX_five_predict
from graphgps.lrx_add.csv_pyg_five_multi import LRX_five_multi
from graphgps.lrx_add.csv_pyg_five_predict_multi import LRX_five_predict_multi


def log_loaded_dataset(dataset, format, name):
    # logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    # logging.info(f"  {dataset.data}")
    # logging.info(f"  undirected: {dataset[0].is_undirected()}")
    # logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
                    or cfg.train.mode == 'double_multi':
                dataset, dataset_2, dataset_3, dataset_4, dataset_5 = preformat_ZINC(dataset_dir, name)
                # dataset, dataset_j = preformat_ZINC(dataset_dir, name)
            else:
                dataset = preformat_ZINC(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    log_loaded_dataset(dataset, format, name)
    if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
            or cfg.train.mode == 'double_multi':
        # log_loaded_dataset(dataset_j, format, name)
        log_loaded_dataset(dataset_2, format, name)
        log_loaded_dataset(dataset_3, format, name)
        log_loaded_dataset(dataset_4, format, name)
        log_loaded_dataset(dataset_5, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        # logging.info(f"Precomputing Positional Encoding statistics: "
        #              f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        # logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
                or cfg.train.mode == 'double_multi':

            pre_transform_in_memory(dataset_2,partial(compute_posenc_stats,pe_types=pe_enabled_list,
                                            is_undirected=is_undirected,
                                            cfg=cfg),show_progress=True)
            pre_transform_in_memory(dataset_3, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                                       is_undirected=is_undirected,
                                                       cfg=cfg), show_progress=True)
            pre_transform_in_memory(dataset_4, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                                       is_undirected=is_undirected,
                                                       cfg=cfg), show_progress=True)
            pre_transform_in_memory(dataset_5, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                                       is_undirected=is_undirected,
                                                       cfg=cfg), show_progress=True)
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')
    if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
            or cfg.train.mode == 'double_multi':
        if hasattr(dataset_2, 'split_idxs'):
            set_dataset_splits(dataset_2, dataset_2.split_idxs)
            delattr(dataset_2, 'split_idxs')
        if hasattr(dataset_3, 'split_idxs'):
            set_dataset_splits(dataset_3, dataset_3.split_idxs)
            delattr(dataset_3, 'split_idxs')
        if hasattr(dataset_4, 'split_idxs'):
            set_dataset_splits(dataset_4, dataset_4.split_idxs)
            delattr(dataset_4, 'split_idxs')
        if hasattr(dataset_5, 'split_idxs'):
            set_dataset_splits(dataset_5, dataset_5.split_idxs)
            delattr(dataset_5, 'split_idxs')
        # if hasattr(dataset_j, 'split_idxs'):
        #     set_dataset_splits(dataset_j, dataset_j.split_idxs)
        #     delattr(dataset_j, 'split_idxs')
    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)
    if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
            or cfg.train.mode == 'double_multi':
        # prepare_splits(dataset_j)
        prepare_splits(dataset_2)
        prepare_splits(dataset_3)
        prepare_splits(dataset_4)
        prepare_splits(dataset_5)

    if cfg.dataset.data_mask == True or cfg.train.mode == 'double' or cfg.train.mode == 'double_predict'\
            or cfg.train.mode == 'double_multi':
        # return dataset, dataset_j
        return dataset, dataset_2, dataset_3, dataset_4, dataset_5
    else:
        return dataset



def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    default_data_path = osp.join(CURRENT_DIR.rsplit('/', 2)[0], 'datasets_lrx')
    data_path = cfg.dataset.dir or default_data_path
    data_path = osp.abspath(data_path)
    if cfg.dataset.cache_per_run:
        source_data_path = data_path
        source_raw = osp.join(source_data_path, 'raw')
        if not osp.lexists(source_raw):
            os.symlink(osp.join(default_data_path, 'raw'), source_raw)
        cache_tag = str(cfg.dataset.cache_tag).strip()
        cache_name = f'{cfg.train.mode}_{cache_tag}_seed_{cfg.seed}' if cache_tag \
            else f'{cfg.train.mode}_seed_{cfg.seed}'
        data_path = osp.join(source_data_path, '.cache', cache_name)
        os.makedirs(data_path, exist_ok=True)
        raw_link = osp.join(data_path, 'raw')
        if not osp.lexists(raw_link):
            os.symlink(source_raw, raw_link)
        print(f"Using isolated dataset cache: {data_path}")
    delete_path = osp.join(data_path, name, 'processed', '*')
    if cfg.dataset.cache_refresh:
        for file in glob.glob(delete_path):
            os.remove(file)
            print("Deleted " + str(file))

    if cfg.dataset.data_mask == True and cfg.train.mode == 'mask':
    ##### own data is suitable for mask subgraph
        data_i_j = [LRX_mask_subgraph(root=data_path, subset=(name == 'subset'), split=split)
             for split in ['train', 'val', 'test','train_j', 'val_j','test_j']]
        dataset = join_dataset_splits(data_i_j[:3])
        dataset_j = join_dataset_splits(data_i_j[3:])
        return dataset, dataset_j


    elif cfg.dataset.data_mask == False and cfg.train.mode == 'double':
        #### five
        if cfg.property_num == 1:
            data_1_5 = [LRX_five(root=data_path, subset=(name == 'subset'), split=split)
                 for split in ['train','val', 'test',
                                     'train_2', 'val_2','test_2',
                                     'train_3','val_3', 'test_3',
                                     'train_4', 'val_4','test_4',
                                     'train_5','val_5','test_5']]
            dataset = join_dataset_splits(data_1_5[:3])
            dataset_2 = join_dataset_splits(data_1_5[3:6])
            dataset_3 = join_dataset_splits(data_1_5[6:9])
            dataset_4 = join_dataset_splits(data_1_5[9:12])
            dataset_5 = join_dataset_splits(data_1_5[12:])
        elif cfg.property_num == 4 or cfg.property_num == 2:
            data_1_5 = [LRX_five_multi(root=data_path, subset=(name == 'subset'), split=split)
                 for split in ['train','val', 'test',
                                     'train_2', 'val_2','test_2',
                                     'train_3','val_3', 'test_3',
                                     'train_4', 'val_4','test_4',
                                     'train_5','val_5','test_5']]
            dataset = join_dataset_splits(data_1_5[:3])
            dataset_2 = join_dataset_splits(data_1_5[3:6])
            dataset_3 = join_dataset_splits(data_1_5[6:9])
            dataset_4 = join_dataset_splits(data_1_5[9:12])
            dataset_5 = join_dataset_splits(data_1_5[12:])
        return dataset, dataset_2, dataset_3, dataset_4, dataset_5


    elif cfg.dataset.data_mask == False and cfg.train.mode == 'double_predict':

        #### five
        if cfg.property_num == 1:
            data_1_5 = [LRX_five_predict(root=data_path, subset=(name == 'subset'), split=split)
                 for split in ['train','val', 'test',
                                     'train_2', 'val_2','test_2',
                                     'train_3','val_3', 'test_3',
                                     'train_4', 'val_4','test_4',
                                     'train_5','val_5','test_5']]
            dataset = join_dataset_splits(data_1_5[:3])
            dataset_2 = join_dataset_splits(data_1_5[3:6])
            dataset_3 = join_dataset_splits(data_1_5[6:9])
            dataset_4 = join_dataset_splits(data_1_5[9:12])
            dataset_5 = join_dataset_splits(data_1_5[12:])
        elif cfg.property_num == 4 or cfg.property_num == 2:
            data_1_5 = [LRX_five_predict_multi(root=data_path, subset=(name == 'subset'), split=split)
                 for split in ['train','val', 'test',
                                     'train_2', 'val_2','test_2',
                                     'train_3','val_3', 'test_3',
                                     'train_4', 'val_4','test_4',
                                     'train_5','val_5','test_5']]
            dataset = join_dataset_splits(data_1_5[:3])
            dataset_2 = join_dataset_splits(data_1_5[3:6])
            dataset_3 = join_dataset_splits(data_1_5[6:9])
            dataset_4 = join_dataset_splits(data_1_5[9:12])
            dataset_5 = join_dataset_splits(data_1_5[12:])
        return dataset, dataset_2, dataset_3, dataset_4, dataset_5




def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
