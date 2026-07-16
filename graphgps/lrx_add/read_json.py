"""
@Name:  json_picture.py
@Auth:  rongxing
@Date:  2024/5/1-上午9:18
@IDE:   PyCharm
@PROJECT_NAME:   $ {PROJECT_NAME}
"""
import jsonlines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read_json(path, picture_list):
    """
    读取jsonl文件，返回loss和6个MAE类指标列表
    """
    Jsonl_Datasets = []
    with open(path, "r+",encoding="utf8") as f:
        for items in jsonlines.Reader(f):
            Jsonl_Datasets.append(items)
    print('------------read is ok')

    loss_list = []
    ee_before_mae_list = []
    ee_after_mae_list = []
    aero_efficiency_mae_list = []
    recovery_efficiency_mae_list = []
    norm_before_mae_list = []
    norm_after_mae_list = []

    for items_ in Jsonl_Datasets:
        loss_list.append(items_['loss'])
        # 读取6个MAE类指标
        ee_before_mae_list.append(items_['mae_per_property']['EE_before_mae'])
        ee_after_mae_list.append(items_['mae_per_property']['EE_after_mae'])
        aero_efficiency_mae_list.append(items_['mae_per_property']['Aero_Efficiency_mae'])
        recovery_efficiency_mae_list.append(items_['mae_per_property']['Recovery_Efficiency_mae'])
        norm_before_mae_list.append(items_['mae_per_property']['Norm_before_mae'])
        norm_after_mae_list.append(items_['mae_per_property']['Norm_after_mae'])

    # 根据picture_list返回对应数据（保持原有逻辑兼容）
    if picture_list == 1:
        # 仅返回loss（loss单独绘图）
        return loss_list
    elif picture_list == 4:
        # 返回所有指标（用于MAE分组绘图）
        return (loss_list, ee_before_mae_list, ee_after_mae_list,
                aero_efficiency_mae_list, recovery_efficiency_mae_list,
                norm_before_mae_list, norm_after_mae_list)
    else:
        return loss_list

def plot_loss_figure(epochs, train_loss, val_loss, test_loss, save_path_loss, num):
    """
    绘制Loss单独的图片
    """
    fig = plt.figure(num=1, figsize=(16, 8))
    ax1 = fig.add_subplot(111)

    # 绘制Loss曲线
    ax1.plot(epochs, train_loss, 'ko', label='Training loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.plot(epochs, test_loss, 'r', label='Test loss')

    # 设置标题和标签
    loss_val_min = min(val_loss)
    loss_best_idx = val_loss.index(loss_val_min)
    ax1.set_title(
        f'Loss - Val Min = {loss_val_min:.3f}, Best Epoch = {loss_best_idx:.0f}'
    )
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 保存图片
    loss_save_path = save_path_loss.replace(f'result_{num}.png', f'loss_result_{num}.png')
    plt.savefig(loss_save_path)
    plt.clf()
    print(f'Loss图片已保存至: {loss_save_path}')

def plot_mae_figures(epochs, train_mae_dict, val_mae_dict, test_mae_dict, save_path_mae, num):
    """
    绘制MAE类指标图片（分两组）
    第一组：EE_before_mae、EE_after_mae
    第二组：Aero_Efficiency_mae、Recovery_Efficiency_mae、Norm_before_mae、Norm_after_mae
    """
    # 第一组MAE：EE相关
    fig1 = plt.figure(num=2, figsize=(16, 8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(epochs, train_mae_dict['EE_before'], 'ko', label='Train EE_before_mae')
    ax1.plot(epochs, val_mae_dict['EE_before'], 'b', label='Val EE_before_mae')
    ax1.plot(epochs, test_mae_dict['EE_before'], 'r', label='Test EE_before_mae')
    ax1.plot(epochs, train_mae_dict['EE_after'], 'mo', label='Train EE_after_mae')
    ax1.plot(epochs, val_mae_dict['EE_after'], 'c', label='Val EE_after_mae')
    ax1.plot(epochs, test_mae_dict['EE_after'], 'g', label='Test EE_after_mae')

    ax1.set_title('MAE - EE_before & EE_after')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ee_save_path = save_path_mae.replace(f'result_{num}.png', f'mae_ee_result_{num}.png')
    plt.savefig(ee_save_path)
    plt.clf()
    print(f'EE MAE图片已保存至: {ee_save_path}')

    # 第二组MAE：其他4个指标
    fig2 = plt.figure(num=3, figsize=(16, 12))
    ax2 = fig2.add_subplot(111)
    # 绘制4个指标曲线
    ax2.plot(epochs, train_mae_dict['Aero_Efficiency'], 'ko', label='Train Aero_Efficiency_mae')
    ax2.plot(epochs, val_mae_dict['Aero_Efficiency'], 'b', label='Val Aero_Efficiency_mae')
    ax2.plot(epochs, test_mae_dict['Aero_Efficiency'], 'r', label='Test Aero_Efficiency_mae')

    ax2.plot(epochs, train_mae_dict['Recovery_Efficiency'], 'mo', label='Train Recovery_Efficiency_mae')
    ax2.plot(epochs, val_mae_dict['Recovery_Efficiency'], 'c', label='Val Recovery_Efficiency_mae')
    ax2.plot(epochs, test_mae_dict['Recovery_Efficiency'], 'g', label='Test Recovery_Efficiency_mae')

    ax2.plot(epochs, train_mae_dict['Norm_before'], 'yo', label='Train Norm_before_mae')
    ax2.plot(epochs, val_mae_dict['Norm_before'], 'orange', label='Val Norm_before_mae')
    ax2.plot(epochs, test_mae_dict['Norm_before'], 'purple', label='Test Norm_before_mae')

    ax2.plot(epochs, train_mae_dict['Norm_after'], 'brown', label='Train Norm_after_mae')
    ax2.plot(epochs, val_mae_dict['Norm_after'], 'pink', label='Val Norm_after_mae')
    ax2.plot(epochs, test_mae_dict['Norm_after'], 'gray', label='Test Norm_after_mae')

    ax2.set_title('MAE - Aero/Recovery/Norm (Before/After)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(alpha=0.3)
    other_save_path = save_path_mae.replace(f'result_{num}.png', f'mae_other_result_{num}.png')
    plt.savefig(other_save_path)
    plt.clf()
    print(f'其他MAE图片已保存至: {other_save_path}')

def result_picture(path_start,read_name,repeat_num,picture_list,metric_best):
    """
    主函数：读取数据并绘制图片
    核心逻辑：以val_loss最小值对应的epoch为最佳模型，提取该epoch下所有指标数值做统计
    """
    # 存储统计结果：每个重复实验中，最佳模型对应的各指标数值
    loss_bestvalid_ckpt_list = []       # 最佳epoch的val_loss
    loss_bestvalid_ckpt_test_list = []  # 最佳epoch的test_loss
    mae_metrics_stats = {
        'EE_before': {'val_best': [], 'test_best': []},
        'EE_after': {'val_best': [], 'test_best': []},
        'Aero_Efficiency': {'val_best': [], 'test_best': []},
        'Recovery_Efficiency': {'val_best': [], 'test_best': []},
        'Norm_before': {'val_best': [], 'test_best': []},
        'Norm_after': {'val_best': [], 'test_best': []}
    }

    for num in range(repeat_num):
        num_path = '/{serial}/'.format(serial =num)
        val_path = path_start + read_name + num_path +'val/stats.json'
        test_path = path_start + read_name + num_path + 'test/stats.json'
        train_path = path_start + read_name + num_path + 'train/stats.json'

        # 读取数据
        if picture_list == 4:
            # 读取loss + 6个MAE指标
            val_data = read_json(val_path, picture_list)
            train_data = read_json(train_path, picture_list)
            test_data = read_json(test_path, picture_list)

            # 解包数据
            val_loss_list, val_EE_before, val_EE_after, val_Aero, val_Recovery, val_Norm_before, val_Norm_after = val_data
            train_loss_list, train_EE_before, train_EE_after, train_Aero, train_Recovery, train_Norm_before, train_Norm_after = train_data
            test_loss_list, test_EE_before, test_EE_after, test_Aero, test_Recovery, test_Norm_before, test_Norm_after = test_data

            # 整理MAE数据为字典（方便调用）
            val_mae_dict = {
                'EE_before': val_EE_before,
                'EE_after': val_EE_after,
                'Aero_Efficiency': val_Aero,
                'Recovery_Efficiency': val_Recovery,
                'Norm_before': val_Norm_before,
                'Norm_after': val_Norm_after
            }
            test_mae_dict = {
                'EE_before': test_EE_before,
                'EE_after': test_EE_after,
                'Aero_Efficiency': test_Aero,
                'Recovery_Efficiency': test_Recovery,
                'Norm_before': test_Norm_before,
                'Norm_after': test_Norm_after
            }

            # 核心：找到val_loss最小值对应的epoch（最佳模型epoch）
            best_epoch = val_loss_list.index(min(val_loss_list))
            print(f'第{num+1}次重复实验 - 最佳模型epoch: {best_epoch} (val_loss最小值: {val_loss_list[best_epoch]:.3f})')

            # 提取最佳epoch下的所有指标数值
            # 1. Loss指标
            loss_bestvalid_ckpt_list.append(val_loss_list[best_epoch])          # 最佳epoch的val_loss
            loss_bestvalid_ckpt_test_list.append(test_loss_list[best_epoch])    # 最佳epoch的test_loss

            # 2. MAE类指标（val和test）
            for key in mae_metrics_stats.keys():
                mae_metrics_stats[key]['val_best'].append(val_mae_dict[key][best_epoch])    # 最佳epoch的val_mae
                mae_metrics_stats[key]['test_best'].append(test_mae_dict[key][best_epoch])  # 最佳epoch的test_mae

            # 生成epochs，绘制图片（原有绘图逻辑不变）
            epochs = range(1, len(train_loss_list) + 1)
            save_path = path_start + read_name + '/' + 'result_'+ str(num) +'.png'
            plot_loss_figure(epochs, train_loss_list, val_loss_list, test_loss_list, save_path, num)
            plot_mae_figures(epochs,
                             {'EE_before':train_EE_before, 'EE_after':train_EE_after, 'Aero_Efficiency':train_Aero,
                              'Recovery_Efficiency':train_Recovery, 'Norm_before':train_Norm_before, 'Norm_after':train_Norm_after},
                             val_mae_dict, test_mae_dict, save_path, num)

        elif picture_list == 1:
            # 仅绘制Loss（兼容原有逻辑）
            val_loss_list = read_json(val_path, picture_list)
            train_loss_list = read_json(train_path, picture_list)
            test_loss_list = read_json(test_path, picture_list)

            # 提取最佳epoch的loss数值
            best_epoch = val_loss_list.index(min(val_loss_list))
            loss_bestvalid_ckpt_list.append(val_loss_list[best_epoch])
            loss_bestvalid_ckpt_test_list.append(test_loss_list[best_epoch])

            epochs = range(1, len(train_loss_list) + 1)
            save_path = path_start + read_name + '/' + 'result_'+ str(num) +'.png'
            plot_loss_figure(epochs, train_loss_list, val_loss_list, test_loss_list, save_path, num)

        print(f'===== 第{num+1}次重复绘制完成 =====')

    # ========== 统计：所有重复实验的最佳模型指标的均值/中位数/标准差 ==========
    # 整理统计结果（同时打印+保存）
    stats_content = []
    stats_content.append(f'总重复实验次数 = {repeat_num}\n')

    # 1. Loss指标统计
    loss_val_ave = np.mean(loss_bestvalid_ckpt_list)
    loss_val_mad = np.mean(np.abs(loss_bestvalid_ckpt_list - loss_val_ave))  # 平均绝对偏差
    loss_val_sd = np.std(loss_bestvalid_ckpt_list)                           # 标准差
    loss_test_ave = np.mean(loss_bestvalid_ckpt_test_list)
    loss_test_mad = np.mean(np.abs(loss_bestvalid_ckpt_test_list - loss_test_ave))
    loss_test_sd = np.std(loss_bestvalid_ckpt_test_list)

    stats_content.append('===== Loss 统计（基于val_loss最佳epoch） =====')
    stats_content.append(f'val_loss_最佳epoch均值 = {loss_val_ave:.3f}, MAD = {loss_val_mad:.3f}, SD = {loss_val_sd:.3f}')
    stats_content.append(f'test_loss_最佳epoch均值 = {loss_test_ave:.3f}, MAD = {loss_test_mad:.3f}, SD = {loss_test_sd:.3f}\n')

    # 2. MAE类指标统计
    stats_content.append('===== MAE 指标统计（基于val_loss最佳epoch） =====')
    for metric, stats in mae_metrics_stats.items():
        val_ave = np.mean(stats['val_best'])
        val_mad = np.mean(np.abs(stats['val_best'] - val_ave))
        val_sd = np.std(stats['val_best'])

        test_ave = np.mean(stats['test_best'])
        test_mad = np.mean(np.abs(stats['test_best'] - test_ave))
        test_sd = np.std(stats['test_best'])

        stats_content.append(f'{metric}_mae:')
        stats_content.append(f'  val_最佳epoch均值 = {val_ave:.3f}, MAD = {val_mad:.3f}, SD = {val_sd:.3f}')
        stats_content.append(f'  test_最佳epoch均值 = {test_ave:.3f}, MAD = {test_mad:.3f}, SD = {test_sd:.3f}\n')

    # 拼接为完整字符串
    stats_str = '\n'.join(stats_content)

    # 打印到控制台
    print('\n' + '='*60 + ' 最终统计结果 ' + '='*60)
    print(stats_str)
    print('='*128 + '\n')

    # 保存到txt文件
    text_path = path_start + read_name + '/result_stats.txt'
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(stats_str)
    print(f'统计结果已保存至: {text_path}')

if __name__ == '__main__':
    metric_best = 'loss'
    read_name = 'direct_layer2_sum_v1_batch4'
    read_path = '/home/lrx/dataset/cooperation/gps/results/'
    repeat_num = 10
    picture_list = 4  # 4=绘制loss+所有MAE指标；1=仅绘制loss
    result_picture(read_path,read_name,repeat_num,picture_list,metric_best)
