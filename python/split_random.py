"""
@Name:  split_random.py
@Auth:  rongxing
@Date:  2026/1/23-上午9:36
@IDE:   PyCharm 
@PROJECT_NAME:   $ {PROJECT_NAME} 
"""
from sklearn.model_selection import train_test_split
import pandas as pd
if __name__ == '__main__':
    data = pd.read_csv('/home/lrx/dataset/cooperation/gps/datasets_lrx/raw/20251215-528-norm.csv')
    train_data, test_data = train_test_split(data, train_size=0.9, test_size=0.1, random_state=int(0))
    train_data, valid_data = train_test_split(train_data, train_size=0.9, test_size=0.1, random_state=int(0))
    valid_data.to_csv('/home/lrx/dataset/cooperation/gps/python/split_csv/seed0_valid.csv')
    test_data.to_csv('/home/lrx/dataset/cooperation/gps/python/split_csv/seed0_test.csv')
    train_data.to_csv('/home/lrx/dataset/cooperation/gps/python/split_csv/seed0_train.csv')
