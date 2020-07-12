import pandas as pd
import copy
import os
from gym_trading.config import PACKAGE_DIR
import numpy as np
from pathlib import Path
from functools import reduce


def normalize_df(df):
    """
    :param df: pandas object, make sure date is lower case
    :return: normalized df
    """
    assert isinstance(df, pd.DataFrame)
    new_df = copy.copy(df)
    new_df.pop('date')
    normalized_df = (new_df - new_df.min()) / (new_df.max() - new_df.min())
    return normalized_df


def construct_df_array(file_array, absolute_path=False):
    """
    :param file_array: array contains file path
    :param absolute_path: whether the file array contains absolute_path
    :return: df final, max share value array of each stock (used for normalization), stock name for reference
    """
    df_array = []
    max_share_value_array = []
    stock_name_array = []
    for file in file_array:
        if absolute_path:
            file_path = file
        else:
            file_path = os.path.join(PACKAGE_DIR, file)

        file_name = Path(file_path).stem
        stock_name = ''.join([c for c in file_name if c.isupper()])
        stock_name_array.append(stock_name)

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()

        df = df.sort_values('date')
        df = df.reset_index(drop=True)  # reset index

        max_share_value = df[['open', 'high', 'low', 'close']].max().max()
        max_share_value_array.append(max_share_value)

        for column_name in df.columns:
            if column_name != 'date':
                target = stock_name + '_' + column_name
                df = df.rename(columns={column_name: target})

        df_array.append(df)

    df_final = reduce(lambda left, right: pd.merge(left, right, on='date'), df_array)

    df_final.dropna  # drop NaN

    return df_final, np.array(max_share_value_array), stock_name_array
