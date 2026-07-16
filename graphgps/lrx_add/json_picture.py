"""
@Name:  json_picture.py
@Auth:  rongxing
@Date:  2024/5/1-上午9:18
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
"""
@Name:  json_R2_picture.py
@Auth:  rongxing
@Date:  2023/1/12-上午10:04
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""

import jsonlines
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read_json(path, picture_list):
    Jsonl_Datasets = []
    with open(path, "r+",encoding="utf8") as f:
        for items in jsonlines.Reader(f):
            Jsonl_Datasets.append(items)
    print('------------read is ok')
    # read  epoch data

    loss_list = []
    mae_list = []
    r2_list = []
    rmse_list = []
    for items_ in Jsonl_Datasets:
        loss_list.append(items_['loss'])
        if picture_list != 1 and picture_list != 2:
            mae_list.append(items_['mae'])
            r2_list.append(items_['r2'])
            rmse_list.append(items_['rmse'])
    loss_list_= loss_list
    # loss_list_ = loss_list[9:]
    if picture_list != 1 and picture_list !=2:
        mae_list_= mae_list #[9:]
        r2_list_= r2_list # [9:]
        rmse_list_= rmse_list #[9:]
    if picture_list != 1 and picture_list != 2 :
        return loss_list_,mae_list_,r2_list_,rmse_list_
    elif picture_list == 1:
        return loss_list_
    else:
        return loss_list_,mae_list_


def result_picture_single(path_start,read_name,repeat_num,picture_list,metric_best):
    if metric_best =='mae':
        result_picture_mae(path_start, read_name, repeat_num, picture_list)
    elif metric_best =='loss':
        result_picture_loss(path_start, read_name, repeat_num, picture_list)

def result_picture_mae(path_start,read_name,repeat_num,picture_list):
    path_start = path_start
    read_path = read_name

    mae_min_ave_list = []
    mae_bestvalid_ckpt_list = []
    mae_bestvalid_ckpt_test_list = []
    for num in range(repeat_num):
        num_path = '/{serial}/'.format(serial =num)
        val_path = path_start + read_path + num_path +'val/stats.json'
        test_path = path_start + read_path + num_path + 'test/stats.json'
        train_path = path_start + read_path + num_path + 'train/stats.json'
        if picture_list == 2:
            print('read_path == zinc-GPS+RWSE_pretrain_mask')
            val_loss_list, val_mae_list = read_json(val_path, picture_list)
            train_loss_list, train_mae_list= read_json(train_path, picture_list)
            test_loss_list, test_mae_list = read_json(test_path, picture_list)
        elif picture_list == 1:
            print('read_path == zinc-GPS+RWSE_pretrain_mask_y')
            val_loss_list = read_json(val_path, picture_list)
            train_loss_list = read_json(train_path, picture_list)
            test_loss_list = read_json(test_path, picture_list)
        else:
            print('else')
            val_loss_list, val_mae_list, val_r2_list, val_rmse_list = read_json(val_path, picture_list)
            train_loss_list, train_mae_list, train_r2_list, train_rmse_list = read_json(train_path, picture_list)
            test_loss_list, test_mae_list, test_r2_list, test_rmse_list = read_json(test_path, picture_list)
        print('-end-')

        if picture_list == 4 :
            best_ckpt_epoch = val_mae_list.index(min(val_mae_list))
            # for seiral,mae_item in enumerate(val_mae_list):
            #     if mae_item == min(val_mae_list):
            #         print('------')
            #         best_ckpt_epoch = seiral
            mae_bestvalid_ckpt_list.append(val_mae_list[best_ckpt_epoch])
            mae_bestvalid_ckpt_test_list.append(test_mae_list[best_ckpt_epoch])
        # epochs = range(10, len(train_loss_list) + 10)
        epochs = range(1, len(train_loss_list) + 1)
        # 创建一个画布
        fig = plt.figure(num=1, figsize=(16, 16))
        # 分成3个子图，一个MSE，一个R, 一个lr
        if picture_list == 1 :
            ax1 = fig.add_subplot(111)
        elif picture_list == 4 :
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224)
        #
        if picture_list == 1 :  #or read_name == 'zinc-GPS+RWSE'
            # ax1.set_ylim(ymin=0, ymax=0.3)
            print('picture_list == 1')
            # ax1.set_ylim(ymin=0, ymax=0.3)
            # ax2.set_ylim(ymin=0, ymax=80)
        else:
            print('-')
            ax2.set_ylim(ymin=0, ymax=80)


        # ax1.plot(epochs, train_loss_list, 'k', label='Training loss')
        # ax1.plot(epochs, val_loss_list, 'b', label='Validation loss')
        # ax1.plot(epochs, test_loss_list, 'r', label='test loss')


        if picture_list == 4:
            ax2.plot(epochs, train_mae_list, 'ko', label='Training mae')
            ax2.plot(epochs, val_mae_list, 'b', label='Validation mae')
            ax2.plot(epochs, test_mae_list, 'r', label='test mae')
            # ax3.set_ylim(ymin=25, ymax=60)
            # ax3.set_title('mae, mae_test_min ={:.3f}'.format(min(test_mae_list))) best_ckpt_epoch
            # ax3.set_title('mae_test_ave ={:.3f},  mae_test_min ={:.3f}'.format(test_mae_list_ave, min(test_mae_list)))
            # ax3.set_title('index={:.1f}'.format(test_mae_list.index(min(test_mae_list))))
            ax2.set_title('mae_valid_min ={:.3f},mae_best_ckpt_test ={:.3f},mae_best_ckpt_index ={:.0f}'.format(min(val_mae_list),
                                                                                            test_mae_list[best_ckpt_epoch],best_ckpt_epoch) + '\n'
                          'mae_test_min ={:.3f}'.format(min(test_mae_list)))
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('mae')
            ax2.legend()
            # mae_ave_list.append(test_mae_list_ave)
            mae_min_ave_list.append(min(test_mae_list))
            ###loss
            ax1.plot(epochs, train_loss_list, 'ko', label='Training loss')
            ax1.plot(epochs, val_loss_list, 'b', label='Validation loss')
            ax1.plot(epochs, test_loss_list, 'r', label='test loss')
            ax1.set_ylim(ymin=0, ymax=80)

            ax1.set_title('loss_val_min ={:.3f} ,loss_best_ckpt_index ={:.0f}'.format(
                    min(val_loss_list), list(val_loss_list).index(
                        min(val_loss_list))))
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('loss')
            ax1.legend()
        elif picture_list == 1:
            ax1.plot(epochs, train_loss_list, 'ko', label='Training loss')
            ax1.plot(epochs, val_loss_list, 'b', label='Validation loss')
            ax1.plot(epochs, test_loss_list, 'r', label='test loss')
            # ax3.set_ylim(ymin=25, ymax=60)
            ax1.set_title(
                'loss_val_min ={:.3f} ,loss_best_ckpt_index ={:.0f}'.format(min(val_loss_list),list(val_loss_list).index(
                                                                                         min(val_loss_list))))
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('loss')
            ax1.legend()
            # mae_ave_list.append(test_mae_list_ave)
            # mae_min_ave_list.append(min(test_mae_list))
        save_path = path_start + read_path + '/' + 'result_'+ str(num) +'.png'
        print('save_path',save_path)
        plt.savefig(save_path)

        # plt.show()
        plt.clf()
        if num == (repeat_num-1) :
            text_path = path_start + read_path + '/result.txt'
            f = open(text_path, 'w')
            if picture_list == 4:
                f.write('sample number = ' + str(num+1) + '\n')
                mae_ave_min = np.mean(mae_min_ave_list)
                f.write(' mae_ave_min = ' + str(mae_ave_min) + '\n')
                print('mae_ave_min', mae_ave_min)
                mae_val_best_ave = np.mean(mae_bestvalid_ckpt_list)
                mae_val_best_test_ave = np.mean(mae_bestvalid_ckpt_test_list)
                # f.write('mae_val_best_ave = ' + str(mae_val_best_ave) + ' mae_val_best_test_ave = ' + str(mae_val_best_test_ave) + '\n')
                # print('mae_val_best_ave', mae_val_best_ave, 'mae_val_best_test_ave', mae_val_best_test_ave)
                mad_mae_val_best_ave = np.mean(np.abs(mae_bestvalid_ckpt_list - mae_val_best_ave))
                mad_mae_val_best_test_ave = np.mean(np.abs(mae_bestvalid_ckpt_test_list - mae_val_best_test_ave))
                mae_val_best_sd = np.std(mae_bestvalid_ckpt_list)
                mae_val_best_test_sd = np.std(mae_bestvalid_ckpt_test_list)
                #### valid_best
                f.write('mae_val_best_ave = ' + str(mae_val_best_ave) + ' mad_mae_val_best_ave = ' + str(
                    mad_mae_val_best_ave) + 'mae_val_best_sd = ' + str(mae_val_best_sd) + '\n')
                print('mae_val_best_ave', mae_val_best_ave, 'mad_mae_val_best_ave', mad_mae_val_best_ave,
                      'mae_val_best_sd = ', mae_val_best_sd)
                #### valid_best_test
                f.write('mae_val_best_test_ave = ' + str(mae_val_best_test_ave) + ' mad_mae_val_best_test_ave = ' + str(
                    mad_mae_val_best_test_ave) + 'mae_val_best_test_sd = ' + str(mae_val_best_test_sd) + '\n')
                print('mae_val_best_test_ave', mae_val_best_test_ave, 'mad_mae_val_best_test_ave',
                      mad_mae_val_best_test_ave,
                      'mae_val_best_test_sd = ', mae_val_best_test_sd)

            elif picture_list == 1:
                f.write(' loss_val_min = ' + str(min(val_loss_list)) + '\n')
                print('mae_ave_min', min(val_loss_list))
                # f.write(' train_loss_list_ave = ' + str(min(val_loss_list)) + 'val_loss_list_ave= '+str(val_loss_list_ave)+'test_loss_list_ave= '+ str(test_loss_list_ave) + '\n')
                # print('mae_ave_min', min(val_loss_list))
            f.close()

def result_picture_loss(path_start,read_name,repeat_num,picture_list):
    path_start = path_start
    read_path = read_name

    mae_bestvalid_ckpt_list = []
    mae_bestvalid_ckpt_test_list = []
    loss_bestvalid_ckpt_list = []
    loss_bestvalid_ckpt_test_list = []
    for num in range(repeat_num):
        num_path = '/{serial}/'.format(serial =num)
        val_path = path_start + read_path + num_path +'val/stats.json'
        test_path = path_start + read_path + num_path + 'test/stats.json'
        train_path = path_start + read_path + num_path + 'train/stats.json'
        if picture_list == 2:
            print('read_path == zinc-GPS+RWSE_pretrain_mask')
            val_loss_list, val_mae_list = read_json(val_path, picture_list)
            train_loss_list, train_mae_list= read_json(train_path, picture_list)
            test_loss_list, test_mae_list = read_json(test_path, picture_list)
        elif picture_list == 1:
            print('read_path == zinc-GPS+RWSE_pretrain_mask_y')
            val_loss_list = read_json(val_path, picture_list)
            train_loss_list = read_json(train_path, picture_list)
            test_loss_list = read_json(test_path, picture_list)
        else:
            print('else')
            val_loss_list, val_mae_list, val_r2_list, val_rmse_list = read_json(val_path, picture_list)
            train_loss_list, train_mae_list, train_r2_list, train_rmse_list = read_json(train_path, picture_list)
            test_loss_list, test_mae_list, test_r2_list, test_rmse_list = read_json(test_path, picture_list)
        print('-end-')

        if picture_list == 4 :
            best_ckpt_epoch = val_loss_list.index(min(val_loss_list))
            mae_bestvalid_ckpt_list.append(val_mae_list[best_ckpt_epoch])
            mae_bestvalid_ckpt_test_list.append(test_mae_list[best_ckpt_epoch])
            loss_bestvalid_ckpt_list.append(val_loss_list[best_ckpt_epoch])
            loss_bestvalid_ckpt_test_list.append(test_loss_list[best_ckpt_epoch])
        # epochs = range(10, len(train_loss_list) + 10)
        epochs = range(1, len(train_loss_list) + 1)
        # 创建一个画布
        fig = plt.figure(num=1, figsize=(16, 16))
        # 分成3个子图，一个MSE，一个R, 一个lr
        if picture_list == 1 :
            ax1 = fig.add_subplot(111)
        elif picture_list == 4 :
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)


        if picture_list == 4:
            ax2.plot(epochs, train_mae_list, 'ko', label='Training mae')
            ax2.plot(epochs, val_mae_list, 'b', label='Validation mae')
            ax2.plot(epochs, test_mae_list, 'r', label='test mae')
            # ax3.set_ylim(ymin=25, ymax=60)

            ax2.set_title('valid_mae_best_ckpt_loss ={:.3f},test_mae_best_ckpt_loss ={:.3f},mae_best_loss_ckpt_index ={:.0f}'.format(
                val_mae_list[val_loss_list.index(min(val_loss_list))],test_mae_list[val_loss_list.index(min(val_loss_list))],val_loss_list.index(min(val_loss_list)))
                          )
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('mae')
            ax2.legend()
            # mae_min_ave_list.append(min(test_mae_list)) ####？？？？？？？？？？？？？？？？
            ###loss
            ax1.plot(epochs, train_loss_list, 'ko', label='Training loss')
            ax1.plot(epochs, val_loss_list, 'b', label='Validation loss')
            ax1.plot(epochs, test_loss_list, 'r', label='test loss')
            # ax1.set_ylim(ymin=0, ymax=300)
            ax1.set_title('loss_val_min ={:.3f} ,loss_best_val_test ={:.3f} ,loss_best_ckpt_index ={:.0f}'.format(
                    val_loss_list[val_loss_list.index(min(val_loss_list))],test_loss_list[val_loss_list.index(min(val_loss_list))],val_loss_list.index(min(val_loss_list))))
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('loss')
            ax1.legend()
        elif picture_list == 1:
            ax1.plot(epochs, train_loss_list, 'ko', label='Training loss')
            ax1.plot(epochs, val_loss_list, 'b', label='Validation loss')
            ax1.plot(epochs, test_loss_list, 'r', label='test loss')
            # ax3.set_ylim(ymin=25, ymax=60)
            ax1.set_title('loss_val_min ={:.3f} ,loss_best_ckpt_index ={:.0f} ,loss_best_ckpt_index ={:.0f}'.format(
                val_loss_list[val_loss_list.index(min(val_loss_list))],
                val_loss_list[val_loss_list.index(min(val_loss_list))], val_loss_list.index(min(val_loss_list))))

            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('loss')
            ax1.legend()
            # mae_ave_list.append(test_mae_list_ave)
            # mae_min_ave_list.append(min(test_mae_list))
        save_path = path_start + read_path + '/' + 'result_'+ str(num) +'.png'
        print('save_path',save_path)
        plt.savefig(save_path)

        # plt.show()
        plt.clf()
        if num == (repeat_num-1) :
            text_path = path_start + read_path + '/result.txt'
            f = open(text_path, 'w')
            if picture_list == 4:
                f.write('sample number = ' + str(num+1) + '\n')

                mae_val_best_ave = np.mean(mae_bestvalid_ckpt_list)
                mae_val_best_test_ave = np.mean(mae_bestvalid_ckpt_test_list)
                mad_mae_val_best_ave = np.mean(np.abs(mae_bestvalid_ckpt_list - mae_val_best_ave))
                mad_mae_val_best_test_ave = np.mean(np.abs(mae_bestvalid_ckpt_test_list - mae_val_best_test_ave))
                mae_val_best_sd = np.std(mae_bestvalid_ckpt_list)
                mae_val_best_test_sd = np.std(mae_bestvalid_ckpt_test_list)
                #### valid_best
                f.write('mae_val_best_ave = ' + str(mae_val_best_ave) + ' mad_mae_val_best_ave = ' + str(
                    mad_mae_val_best_ave) + 'mae_val_best_sd = ' + str(mae_val_best_sd) + '\n')
                print('mae_val_best_ave', mae_val_best_ave, 'mad_mae_val_best_ave', mad_mae_val_best_ave,
                      'mae_val_best_sd = ', mae_val_best_sd)
                #### valid_best_test
                f.write('mae_val_best_test_ave = ' + str(mae_val_best_test_ave) + ' mad_mae_val_best_test_ave = ' + str(
                    mad_mae_val_best_test_ave) + 'mae_val_best_test_sd = ' + str(mae_val_best_test_sd) + '\n')
                print('mae_val_best_test_ave', mae_val_best_test_ave, 'mad_mae_val_best_test_ave',
                      mad_mae_val_best_test_ave,
                      'mae_val_best_test_sd = ', mae_val_best_test_sd)

                loss_val_best_ave = np.mean(loss_bestvalid_ckpt_list)
                loss_val_best_test_ave = np.mean(loss_bestvalid_ckpt_test_list)
                # f.write('loss_val_best_ave = ' + str(loss_val_best_ave) + ' loss_val_best_test_ave = ' + str(
                #     loss_val_best_test_ave) + '\n')
                # print('loss_val_best_ave', loss_val_best_ave, 'loss_val_best_test_ave', loss_val_best_test_ave)
            elif picture_list == 1:

                loss_val_best_ave = np.mean(loss_bestvalid_ckpt_list)
                loss_val_best_test_ave = np.mean(loss_bestvalid_ckpt_test_list)
                f.write('loss_val_best_ave = ' + str(loss_val_best_ave) + ' loss_val_best_test_ave = ' + str(
                    loss_val_best_test_ave) + '\n')
                print('loss_val_best_ave', loss_val_best_ave, 'loss_val_best_test_ave', loss_val_best_test_ave)

            f.close()

if __name__ == '__main__':
    metric_best = 'loss'
    read_name = 'direct_layer6_batch4_multi4_ratio_weighted'
    read_path = '/home/lrx/dataset/cooperation/gps/results/'
    # read_name = 'mask_seed_1_layer2_v41_cat_DP'
    # read_path = '/home/lrx/dataset/polymer_code/poly_gps_double/results/mask/DP/'
    repeat_num = 10
    picture_list = 4
    result_picture_single(read_path,read_name,repeat_num,picture_list,metric_best)
