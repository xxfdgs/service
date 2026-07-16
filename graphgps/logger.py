import logging
import time

import numpy as np
import torch
from scipy.stats import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    confusion_matrix
from sklearn.metrics import r2_score
from torch_geometric.graphgym import get_current_gpu_usage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.logger import infer_task, Logger
from torch_geometric.graphgym.utils.io import dict_to_json, dict_to_tb
from torchmetrics.functional import auroc

import graphgps.metrics_ogb as metrics_ogb
from graphgps.metric_wrapper import MetricWrapper


def accuracy_SBM(targets, pred_int):
    """Accuracy eval for Benchmarking GNN's PATTERN and CLUSTER datasets.
    https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/train/metrics.py#L34
    """
    S = targets
    C = pred_int
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = np.sum(pr_classes) / float(nb_classes)
    return acc


class CustomLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False

    # basic properties
    def basic(self):
        stats = {
            'loss': round(self._loss / self._size_current, max(8, cfg.round)),
            'lr': round(self._lr, max(8, cfg.round)),
            'params': self._params,
            'time_iter': round(self.time_iter(), cfg.round),
        }
        gpu_memory = get_current_gpu_usage()
        if gpu_memory > 0:
            stats['gpu_memory'] = gpu_memory
        return stats

    # task properties
    def classification_binary(self):
        true = torch.cat(self._true).squeeze(-1)
        pred_score = torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)

        if true.shape[0] < 1e7:  # AUROC computation for very large datasets is too slow.
            # TorchMetrics AUROC on GPU if available.
            auroc_score = auroc(pred_score.to(torch.device(cfg.device)),
                                true.to(torch.device(cfg.device)),
                                pos_label=1)
            if self.test_scores:
                # SK-learn version.
                try:
                    r_a_score = roc_auc_score(true.cpu().numpy(),
                                              pred_score.cpu().numpy())
                except ValueError:
                    r_a_score = 0.0
                assert np.isclose(float(auroc_score), r_a_score)
        else:
            auroc_score = 0.

        reformat = lambda x: round(float(x), cfg.round)
        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'precision': reformat(precision_score(true, pred_int)),
            'recall': reformat(recall_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int)),
            'auc': reformat(auroc_score),
        }
        if cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))
        return res

    def classification_multi(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        pred_int = self._get_pred_int(pred_score)
        reformat = lambda x: round(float(x), cfg.round)

        res = {
            'accuracy': reformat(accuracy_score(true, pred_int)),
            'f1': reformat(f1_score(true, pred_int,
                                    average='macro', zero_division=0)),
        }
        if cfg.metric_best == 'accuracy-SBM':
            res['accuracy-SBM'] = reformat(accuracy_SBM(true, pred_int))
        if true.shape[0] < 1e7:
            # AUROC computation for very large datasets runs out of memory.
            # TorchMetrics AUROC on GPU is much faster than sklearn for large ds
            res['auc'] = reformat(auroc(pred_score.to(torch.device(cfg.device)),
                                        true.to(torch.device(cfg.device)).squeeze(),
                                        num_classes=pred_score.shape[1],
                                        average='macro'))

            if self.test_scores:
                # SK-learn version.
                sk_auc = reformat(roc_auc_score(true, pred_score.exp(),
                                                average='macro',
                                                multi_class='ovr'))
                assert np.isclose(sk_auc, res['auc'])

        return res

    def classification_multilabel(self):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)

        # Send to GPU to speed up TorchMetrics if possible.
        true = true.to(torch.device(cfg.device))
        pred_score = pred_score.to(torch.device(cfg.device))
        acc = MetricWrapper(metric='accuracy',
                            target_nan_mask='ignore-mean-label',
                            threshold=0.,
                            cast_to_int=True)
        ap = MetricWrapper(metric='averageprecision',
                           target_nan_mask='ignore-mean-label',
                           pos_label=1,
                           cast_to_int=True)
        auroc = MetricWrapper(metric='auroc',
                              target_nan_mask='ignore-mean-label',
                              pos_label=1,
                              cast_to_int=True)
        results = {
            'accuracy': reformat(acc(pred_score, true)),
            'ap': reformat(ap(pred_score, true)),
            'auc': reformat(auroc(pred_score, true)),
        }

        if self.test_scores:
            # Compute metric by OGB Evaluator methods.
            true = true.cpu().numpy()
            pred_score = pred_score.cpu().numpy()
            ogb = {
                'accuracy': reformat(metrics_ogb.eval_acc(
                    true, (pred_score > 0.).astype(int))['acc']),
                'ap': reformat(metrics_ogb.eval_ap(true, pred_score)['ap']),
                'auc': reformat(
                    metrics_ogb.eval_rocauc(true, pred_score)['rocauc']),
            }
            assert np.isclose(ogb['accuracy'], results['accuracy'])
            assert np.isclose(ogb['ap'], results['ap'])
            assert np.isclose(ogb['auc'], results['auc'])

        return results

    def subtoken_prediction(self):
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator('ogbg-code2')

        seq_ref_list = []
        seq_pred_list = []
        for seq_pred, seq_ref in zip(self._pred, self._true):
            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

        input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
        result = evaluator.eval(input_dict)
        result['f1'] = result['F1']
        del result['F1']
        return result

    ### single property
    def regression(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)

        if cfg.dataset.data_mask == True and cfg.train.mode != 'double':
            return {
            }
        elif cfg.train.mode == 'double':
            if cfg.property_num == 1: ### single property
                return {
                    'mae': reformat(mean_absolute_error(true, pred)),
                    'r2': reformat(r2_score(true, pred, multioutput='uniform_average')),
                    'rmse': reformat(mean_squared_error(true, pred)),
                }
            elif cfg.property_num == 6: #### multipli property
                # 转换为numpy数组（sklearn计算更方便）
                true_np = true.numpy()
                pred_np = pred.numpy()

                # 1. 计算每个性质单独的MAE
                mae_per_property = {}
                num_properties = true_np.shape[1]  # 自动识别6个性质
                ###
                # 第一步：定义你的自定义性质名称列表（放在循环外）
                property_names = ['EE_before', 'EE_after', 'Aero_Efficiency', 'Recovery_Efficiency',
                                  'Norm_before', 'Norm_after']

                # 第二步：修改循环逻辑，用自定义名称替代数字索引
                for prop_idx in range(num_properties):
                    # 提取第prop_idx个性质的所有样本数据
                    true_prop = true_np[:, prop_idx]
                    pred_prop = pred_np[:, prop_idx]
                    # 关键修改：用property_names[prop_idx]获取自定义名称
                    prop_name = property_names[prop_idx]
                    # 拼接MAE后缀，生成最终键名（如'EE_before_mae'）
                    mae_per_property[f'{prop_name}_mae'] = reformat(mean_absolute_error(true_prop, pred_prop))
                ###

                # 2. 计算6个性质的平均MAE（和原代码multioutput逻辑一致）
                mae_average = reformat(mean_absolute_error(true, pred, multioutput='uniform_average'))

                # 返回结果：既包含每个性质的MAE，也包含整体平均MAE
                return {
                    'mae_per_property': mae_per_property,  # 每个性质的MAE（6个）
                    'mae': mae_average  # 6个性质的平均MAE
                }
            elif cfg.property_num == 4: #### multipli property
                true_np= true.numpy()
                pred_np = pred.numpy()
                PROPERTY_NUM = 4
                batch_size = len(true_np) // PROPERTY_NUM

                # 还原为 [batch_size, 4]（对应“先batch再性质”的平铺方式）
                true_2d = true_np.reshape(batch_size, PROPERTY_NUM)
                pred_2d = pred_np.reshape(batch_size, PROPERTY_NUM)

                # 每个性质的 MAE
                property_names = ['EE_before', 'EE_after', 'Aero_Efficiency', 'Recovery_Efficiency']
                mae_per_property = {}
                mae_list = []

                for prop_idx in range(PROPERTY_NUM):
                    true_prop = true_2d[:, prop_idx]
                    pred_prop = pred_2d[:, prop_idx]

                    prop_name = property_names[prop_idx]
                    mae_i = mean_absolute_error(true_prop, pred_prop)  # 标量
                    mae_per_property[f'{prop_name}_mae'] = reformat(mae_i)
                    mae_list.append(mae_i)

                # 4个性质 MAE 加合（你要的）
                mae_sum = reformat(float(np.sum(mae_list)))

                return {
                    'mae_per_property': mae_per_property,  # 每个性质的MAE
                    'mae_sum': mae_sum  # MAE加合
                }
            elif cfg.property_num == 2: #### multipli property
                true_np= true.numpy()
                pred_np = pred.numpy()
                PROPERTY_NUM = 2
                batch_size = len(true_np) // PROPERTY_NUM

                # 还原为 [batch_size, 4]（对应“先batch再性质”的平铺方式）
                true_2d = true_np.reshape(batch_size, PROPERTY_NUM)
                pred_2d = pred_np.reshape(batch_size, PROPERTY_NUM)

                # 每个性质的 MAE
                property_names = ['Norm_before', 'Norm_after']
                mae_per_property = {}
                mae_list = []

                for prop_idx in range(PROPERTY_NUM):
                    true_prop = true_2d[:, prop_idx]
                    pred_prop = pred_2d[:, prop_idx]

                    prop_name = property_names[prop_idx]
                    mae_i = mean_absolute_error(true_prop, pred_prop)  # 标量
                    mae_per_property[f'{prop_name}_mae'] = reformat(mae_i)
                    mae_list.append(mae_i)

                # 4个性质 MAE 加合（你要的）
                mae_sum = reformat(float(np.sum(mae_list)))

                return {
                    'mae_per_property': mae_per_property,  # 每个性质的MAE
                    'mae_sum': mae_sum  # MAE加合
                }
    def regression_out(self):
        true, pred = torch.cat(self._true), torch.cat(self._pred)
        reformat = lambda x: round(float(x), cfg.round)

        if cfg.dataset.data_mask == True and cfg.train.mode != 'double':
            return {
            }
        elif cfg.train.mode == 'double':
            if cfg.property_num == 1: ### single property
                return {
                    'mae': reformat(mean_absolute_error(true, pred)),
                    'r2': reformat(r2_score(true, pred, multioutput='uniform_average')),
                    'rmse': reformat(mean_squared_error(true, pred)),
                }
            elif cfg.property_num == 6: #### multipli property
                # 转换为numpy数组（sklearn计算更方便）
                true_np = true.numpy()
                pred_np = pred.numpy()

                # 1. 计算每个性质单独的MAE
                mae_per_property = {}
                num_properties = true_np.shape[1]  # 自动识别6个性质
                ###
                # 第一步：定义你的自定义性质名称列表（放在循环外）
                property_names = ['EE_before', 'EE_after', 'Aero_Efficiency', 'Recovery_Efficiency',
                                  'Norm_before', 'Norm_after']

                # 第二步：修改循环逻辑，用自定义名称替代数字索引
                for prop_idx in range(num_properties):
                    # 提取第prop_idx个性质的所有样本数据
                    true_prop = true_np[:, prop_idx]
                    pred_prop = pred_np[:, prop_idx]
                    # 关键修改：用property_names[prop_idx]获取自定义名称
                    prop_name = property_names[prop_idx]
                    # 拼接MAE后缀，生成最终键名（如'EE_before_mae'）
                    mae_per_property[f'{prop_name}_mae'] = reformat(mean_absolute_error(true_prop, pred_prop))
                ###

                # 2. 计算6个性质的平均MAE（和原代码multioutput逻辑一致）
                mae_average = reformat(mean_absolute_error(true, pred, multioutput='uniform_average'))

                # 返回结果：既包含每个性质的MAE，也包含整体平均MAE
                return {
                    'mae_per_property': mae_per_property,  # 每个性质的MAE（6个）
                    'mae': mae_average  # 6个性质的平均MAE
                }
            elif cfg.property_num == 4: #### multipli property
                true_np= true.numpy()
                pred_np = pred.numpy()
                PROPERTY_NUM = 4
                batch_size = len(true_np) // PROPERTY_NUM

                # 还原为 [batch_size, 4]（对应“先batch再性质”的平铺方式）
                true_2d = true_np.reshape(batch_size, PROPERTY_NUM)
                pred_2d = pred_np.reshape(batch_size, PROPERTY_NUM)

                # 每个性质的 MAE
                property_names = ['EE_before', 'EE_after', 'Aero_Efficiency', 'Recovery_Efficiency']
                mae_per_property = {}
                mae_list = []

                for prop_idx in range(PROPERTY_NUM):
                    true_prop = true_2d[:, prop_idx]
                    pred_prop = pred_2d[:, prop_idx]

                    prop_name = property_names[prop_idx]
                    mae_i = mean_absolute_error(true_prop, pred_prop)  # 标量
                    mae_per_property[f'{prop_name}_mae'] = reformat(mae_i)
                    mae_list.append(mae_i)

                # 4个性质 MAE 加合（你要的）
                mae_sum = reformat(float(np.sum(mae_list)))

                return {
                    'mae_per_property': mae_per_property,  # 每个性质的MAE
                    'mae_sum': mae_sum  # MAE加合
                }
            elif cfg.property_num == 2: #### multipli property
                true_np= true.numpy()
                pred_np = pred.numpy()
                PROPERTY_NUM = 2
                batch_size = len(true_np) // PROPERTY_NUM

                # 还原为 [batch_size, 4]（对应“先batch再性质”的平铺方式）
                true_2d = true_np.reshape(batch_size, PROPERTY_NUM)
                pred_2d = pred_np.reshape(batch_size, PROPERTY_NUM)

                # 每个性质的 MAE
                property_names = ['Norm_before', 'Norm_after']
                mae_per_property = {}
                mae_list = []

                for prop_idx in range(PROPERTY_NUM):
                    true_prop = true_2d[:, prop_idx]
                    pred_prop = pred_2d[:, prop_idx]

                    prop_name = property_names[prop_idx]
                    mae_i = mean_absolute_error(true_prop, pred_prop)  # 标量
                    mae_per_property[f'{prop_name}_mae'] = reformat(mae_i)
                    mae_list.append(mae_i)

                # 4个性质 MAE 加合（你要的）
                mae_sum = reformat(float(np.sum(mae_list)))

                return {
                    'mae_per_property': mae_per_property,  # 每个性质的MAE
                    'mae_sum': mae_sum  # MAE加合
                }
    def update_stats(self, true, pred, loss, lr, time_used, params,
                     dataset_name=None, **kwargs):
        if dataset_name == 'ogbg-code2':
            assert true['y_arr'].shape[1] == len(pred)  # max_seq_len (5)
            assert true['y_arr'].shape[0] == pred[0].shape[0]  # batch size
            batch_size = true['y_arr'].shape[0]

            # Decode the predicted sequence tokens, so we don't need to store
            # the logits that take significant memory.
            from graphgps.loader.ogbg_code2_utils import idx2vocab, \
                decode_arr_to_seq
            arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)
            mat = []
            for i in range(len(pred)):
                mat.append(torch.argmax(pred[i].detach(), dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)
            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [true['y'][i] for i in range(len(true['y']))]
            pred = seq_pred
            true = seq_ref
        else:
            assert true.shape[0] == pred.shape[0]
            batch_size = true.shape[0]
        self._iter += 1
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._lr = lr
        self._params = params
        self._time_used += time_used
        self._time_total += time_used
        for key, val in kwargs.items():
            if key not in self._custom_stats:
                self._custom_stats[key] = val * batch_size
            else:
                self._custom_stats[key] += val * batch_size
    def out_predict(self):
        true_best, pred_best = self.regression_out()
        return true_best, pred_best
    def write_epoch(self, cur_epoch):
        start_time = time.perf_counter()
        basic_stats = self.basic()

        if self.task_type == 'regression':
            task_stats = self.regression()
        else:
            raise ValueError('Task has to be regression or classification')

        epoch_stats = {'epoch': cur_epoch,
                       'time_epoch': round(self._time_used, cfg.round)}
        eta_stats = {'eta': round(self.eta(cur_epoch), cfg.round),
                     'eta_hours': round(self.eta(cur_epoch) / 3600, cfg.round)}
        custom_stats = self.custom()

        if self.name == 'train':
            stats = {
                **epoch_stats,
                **eta_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }
        else:
            stats = {
                **epoch_stats,
                **basic_stats,
                **task_stats,
                **custom_stats
            }

        # print
        logging.info('{}: {}'.format(self.name, stats))
        # json
        dict_to_json(stats, '{}/stats.json'.format(self.out_dir))
        # tensorboard
        if cfg.tensorboard_each_run:
            dict_to_tb(stats, self.tb_writer, cur_epoch)
        self.reset()
        if cur_epoch < 3:
            logging.info(f"...computing epoch stats took: "
                         f"{time.perf_counter() - start_time:.2f}s")
        return stats


def create_logger():
    """
    Create logger for the experiment

    Returns: List of logger objects

    """
    loggers = []
    # names = ['train', 'val', 'test']
    ##lrx add
    if cfg.train.mode == 'core':
        names = ['train_similar', 'train_core', 'val', 'test']
    else:
        names = ['train', 'val', 'test']
    for i, dataset in enumerate(range(cfg.share.num_splits)):
        loggers.append(CustomLogger(name=names[i], task_type=infer_task()))
    return loggers


def eval_spearmanr(y_true, y_pred):
    """Compute Spearman Rho averaged across tasks.
    """
    res_list = []

    if y_true.ndim == 1:
        res_list.append(stats.spearmanr(y_true, y_pred)[0])
    else:
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = ~np.isnan(y_true[:, i])
            res_list.append(stats.spearmanr(y_true[is_labeled, i],
                                            y_pred[is_labeled, i])[0])

    return {'spearmanr': sum(res_list) / len(res_list)}
