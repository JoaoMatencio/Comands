from typing import List, Dict, Tuple, Union
import pandas as pd
import scipy.stats as stats
import numpy as np


def is_numeric_value(value: Union[str, int, float]) -> bool:
    """Take a value and Verify if the value is numeric and return True or False

    Arguments:
        value: Any type of string you want to check

    Returns:
        If the value is True or False
    """
    value = str(value)
    if value == "nan":
        return True
    return all([c.isdigit() or c in ".,R$\xa0 " and c not in "/-" for c in value])


def is_numeric_col(col: pd.Series) -> bool:
    """Take a col and Verify if the col is numeric and return True or False

    Arguments:
        col: Column you want to verify

    Returns:
        If the col is True or False
    """
    return all(col.apply(is_numeric_value))


def is_numeric_df(df: pd.DataFrame) -> List[str] and List[str]:
    """Receives a df and checks numeric and non-numeric columns.

    Arguments:
        df: original dataset to verify if is numeric

    Returns:
        numeric_cols: The numeric's cols
        cols_not_numeric: The not numeric's cols
    """
    numeric_cols = []
    cols_not_numeric = []
    for i in df:
        if is_numeric_col(df[i]):
            numeric_cols.append(i)
        else:
            cols_not_numeric.append(i)
    return numeric_cols, cols_not_numeric


def is_normal(df: pd.DataFrame, numeric_cols: list, alpha: float) -> bool:
    """Verify if the distribuicion is normal

    Arguments:
        df: original dataset to be plot
        numeric_cols:    the list of the numerics series
        alpha: Confiance level
    Returns:
        List of columns that have normal distribution.
    """
    normal_cols = []
    for i in df:
        if i in numeric_cols:
            W, p = stats.shapiro(df[i])
            if p > alpha:
                normal_cols.append(i)
    return normal_cols


def is_exponential(df: pd.DataFrame, numeric_cols: list, alpha: float) -> bool:
    """Verify if the distribution is exponential

    Arguments:
        df: original dataset to be plot
        numeric_cols:    the list of the numerics series
        alpha: Confiance level
    Returns:
        List of columns that have exponential distribution.
    """
    exponencial_cols = []
    for i in df:
        if i in numeric_cols:
            loc, scale = stats.expon.fit(df[i], floc=0)
            D, p = stats.kstest(df[i], "expon", args=(loc, scale))
            if p > alpha:
                exponencial_cols.append(i)
    return exponencial_cols


# def plot_hexbin(df, x, y, grid, color, bins, **params):
#    args = {'x':x, 'y': y, 'grid': grid, 'color': color}
#    if bins == 'log':
#        args['bins'] = 'log'
#    df.plot.hexbin(**args, **params)
