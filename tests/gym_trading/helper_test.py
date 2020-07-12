import os
import pytest
import pandas as pd
from gym_trading.envs.helper import construct_df_array
from gym_trading.config import PACKAGE_DIR


def test_construct_df_array():
    file_array = ['data/daily_IBM.csv']
    df_final, max_share_value_array, stock_name_array = construct_df_array(file_array)
    for i, file, stock_name in zip(range(len(file_array)), file_array, stock_name_array):
        file_path = os.path.join(PACKAGE_DIR, file)
        df = pd.read_csv(file_path)
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        for column_name in df.columns:
            if column_name != 'date':
                target = stock_name + '_' + column_name
                df = df.rename(columns={column_name: target})
        assert ((df_final == df).all()).all()
