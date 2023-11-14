"""
Implementing a SVM ML technique.
Part One. Import and clean data.

Author: Julia De Geest
"""
import pandas as pd
from scipy.io import arff


def clean_data(filename: str) -> 'pandas.df':
    data = arff.loadarff(filename)

    df = pd.DataFrame(data[0])

    # change attribute types from bytes to ints
    for col in df:
        df[col] = [int(i) for i in df[col]]

    # if data points are missing a value, drop the row. taking the average value wouldn't work here because 1) the data
    # is continuous so it would have to be the median and 2) it might mess up the classification weights because it's
    # result wouldn't correlate directly
    nan_rows = list(df.index[df.isna().any(axis=1)])
    if not nan_rows:
        for i in range(len(nan_rows)):
            df.drop(nan_rows[i], inplace=True)

    return df
