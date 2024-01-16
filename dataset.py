import pandas as pd
import numpy as np

def replace_missing_with_median_numeric(dataframe):
    # Select only numeric columns
    numeric_columns = dataframe.select_dtypes(include=['number']).columns

    # Iterate through each numeric column
    for column in numeric_columns:
        # Calculate the median excluding missing values
        median_value = dataframe[column].median()

        # Replace missing values in the numeric column with the median
        dataframe[column].fillna(median_value, inplace=True)

    return dataframe

def treat_dataset(df):
    pool_quality_mapping = {'Ex': 400, 'Gd': 300, 'TA': 200, 'Fa': 100, 'NA': 0}
    df['PoolQC_encoded'] = df['PoolQC'].map(pool_quality_mapping)
    df.drop('PoolQC', axis=1, inplace=True)

    pool_quality_mapping = {'GdPrv': 400, 'MnPrv': 300, 'GdWo': 200, 'MnWw': 100, 'NA': 0}
    df['Fence_encoded'] = df['Fence'].map(pool_quality_mapping)
    df.drop('Fence', axis=1, inplace=True)

    pool_quality_mapping = {'Y': 200, 'P': 100, 'N': 10}
    df['PavedDrive_encoded'] = df['PavedDrive'].map(pool_quality_mapping)
    df.drop('PavedDrive', axis=1, inplace=True)

    df = replace_missing_with_median_numeric(df)
    return df
