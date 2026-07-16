"""
@Name:  predict_average.py
@Auth:  rongxing
@Date:  2023/4/25-下午4:04
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME}
average predicited datas from different pretrain_finetuning_models
and analysis function of average
output pictures and csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

def read_data(serial,file_name,save_path,true_num):
    read_path = save_path + file_name+'/' + str(
        serial) + 'test_true_pred_sum.csv'
    read_data = pd.read_csv(read_path, index_col=0)[:true_num]
    # print('len(read_data)',len(read_data))
    return read_data

def se(true_list,predict_list):
    se_ = np.std(np.array((predict_list)-np.array(true_list))/np.sqrt(len(predict_list)))
    se = se_ / 1
    return se

def save_picture(path,csv,csv_file,serial_num):
    fig, ax1 = plt.subplots(figsize=(8, 8))

    fontsize = 20
    #data
    pred_value = csv['average predict']
    true_value = csv['true value']
    criterion_list = []
    if serial_num == 5 or serial_num ==4:
        for item in range(0, 5):
            criterion_list.append(item)
    else:
        for item in range(0, 101):
            criterion_list.append(item)

    plt.plot(criterion_list, criterion_list, markersize=3, alpha=0.8, label='true',
             linewidth=1.5)
    # plt.plot(true_value, true_value, markersize=3, alpha=0.8, label='true',
    #          linewidth=1.5)
    plt.plot(true_value, pred_value, 'r+', markersize=8, alpha=0.8, label='pre')

    r_2 = r2_score(true_value, pred_value)
    MAE = mean_absolute_error(true_value, pred_value)
    RMSE = mean_squared_error(true_value, pred_value)
    print(r_2,MAE,RMSE)
    # title_label = ' MAE = {:.1f}m  RMSE = {:.1f}m  R\N{SUPERSCRIPT TWO}={:.3f}'.format(MAE, RMSE,
    #                                                                                    r_2)
    # ax1.set_ylim(ymin=125, ymax=525)
    if serial_num ==0:
        title_label = 'EE_before MAE = {:.2f}℃'.format(MAE)
    elif serial_num ==1:
        title_label = 'EE_after MAE = {:.2f}℃'.format(MAE)
    elif serial_num ==2:
        title_label = 'Aero_Efficiency MAE = {:.2f}℃'.format(MAE)
    elif serial_num ==3:
        title_label = 'Recovery_Efficiency MAE = {:.2f}℃'.format(MAE)
    elif serial_num ==4:
        title_label = 'Norm_before MAE = {:.2f}℃'.format(MAE)
    elif serial_num ==5:
        title_label = 'Norm_after MAE = {:.2f}℃'.format(MAE)
    print('PN MAE =', MAE)
    # title_label = ' MAE = {:.1f} ,R2 = {:.2f} '.format(MAE, r_2)
    fig.suptitle(title_label, fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax1.set_ylim(ymin=100, ymax=500)
    # ax1.set_xlim(xmin=100, xmax=500)

    ax1.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax1.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax1.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax1.spines['top'].set_linewidth(2);  ####设置上部坐标轴的粗细
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Experiment', fontsize=14)
    plt.ylabel('Model-prediction', fontsize=14)
    # save data by the csv and picture


    plt_save_name = 'predict_average' + '.png'
    plt.savefig(path+plt_save_name, dpi=100)

    plt.close()

def average_accuracy(start_list):
    true_list = start_list['true value']
    average_list = start_list['average predict']
    result_list = []
    for serial,item in enumerate(true_list):
        # cha_= abs(1 - (average_list[serial]/item))
        cha_ = abs(item - average_list[serial])/item
        result_list.append(cha_)
    print('average accuracy=',np.mean(result_list))

def main_run_single(file_name,save_path,loop_num,csv_file,serial_num):

    # loop_num = 10
    model_list = []
    read_csv = pd.read_csv(csv_file)
    read_num = len(read_csv['IL_SMILE'])
    # read csv
    for serial in range(loop_num):
        if serial == 0:
            start_list = read_data(serial, file_name,save_path,read_num)
            start_list.rename(columns={'pred value': 'pred value model 0'}, inplace=True)
            new_name = 'pred value model ' + str(serial)
            model_list.append(new_name)
        else:
            new_list = read_data(serial, file_name,save_path,read_num)
            old_name = 'pred value'
            new_name = 'pred value model ' + str(serial)
            model_list.append(new_name)
            new_list.rename(columns={old_name: new_name}, inplace=True)
            start_list = pd.concat([start_list, new_list], axis=1)

    start_list = start_list.loc[:, ~start_list.columns.duplicated()]

    # column_index = [item for item in range(2,2+loop_num)]
    list_row = []
    for index, row in start_list.iterrows():
        count = 0
        sum_ = 0.0
        # print('count=',count,'sum_=',sum_)
        for serial, item_row in enumerate(row):
            if serial >= 1:
                count += 1
                sum_ += item_row
                # print('count=', count, 'sum_=', sum_)
        ave = sum_ / count
        list_row.append(round(ave,2))
    start_list['average predict'] = list_row
    model_list.append('average predict')

    save_picture_path = save_path + file_name + '/'
    # save_picture_path = '/home/lrx/dataset/code/GraphGPS-main/results/predict/s2/'
    if 'tem_generate' not in csv_file:
        save_picture(save_picture_path, start_list,csv_file,serial_num)
    # ####
    mae_list = []
    se_list = []
    r2_list = []
    for model_item in model_list:
        # if model_item != 'error':
        pred_value = start_list[model_item]
        true_value = start_list['true value']
        MAE_ = mean_absolute_error(true_value, pred_value)
        R2_ = r2_score(true_value, pred_value)
        se_ = se(true_value, pred_value)
        se_list.append(se_)
        mae_list.append(MAE_)
        r2_list.append(R2_)

    print('R2= ',r2_score(start_list['true value'],start_list['average predict']))
    average_accuracy(start_list)
    true_list = start_list['true value']
    true_list['true'] = true_list
    start_list_ = start_list.round(2)
    start_list_.to_csv(save_path + file_name + '/predicted_average.csv')

if __name__ == '__main__':
    loop_num = 10
    model_list = []
    read_csv = '/home/lrx/dataset/cooperation/gps/datasets_lrx/raw/feedback/20260116_Prediction_7_96_top_center_bottom.csv'
    file_name = 'direct_layer3_batch4_single_cat_v1_list5_Norm_after'
    save_path = '/home/lrx/dataset/cooperation/gps/results/predict/single_property/cat/list5_Norm_after/v1/'
    serial_num = 5
    main_run_single(file_name, save_path,loop_num,read_csv,serial_num)
    print('----- all over -----')
