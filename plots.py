import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numeric import NumericFunctions as nf


def plotar_pizza(df: pd.DataFrame, cols_not_numeric: list, threshold: float) -> None:
    """Take the chosen df and limit and plot a pie chart of the non-numeric columns

    Arguments:
        df:         original dataset to be plot
        threshold:  defines how many classes will be shown on the pizza plot
    """
    for i in df:
        if i in cols_not_numeric:
            contagem_categorias = df[i].value_counts()
            limite = contagem_categorias < threshold
            outros = contagem_categorias.loc[limite]
            contagem_categorias = contagem_categorias.loc[~limite]
            contagem_categorias["Outros"] = outros.sum()
            contagem_categorias.plot(kind="pie", autopct="%1.1f%%", startangle=90)
            plt.title(i)
            plt.show()


def plot(df: pd.DataFrame, numeric_cols: list, bins: int) -> None:
    """Take the df and the chosen bins and plot a histogram of the numeric columns

    Arguments:
        df:             original dataset to be plot
        numeric_cols:    the list of the numerics series
        bins:           bins chosen for the plot
    """
    for i in df:
        if i in numeric_cols:
            plt.hist(df[i], bins, color="blue", edgecolor="black")
            plt.title(i)
            plt.xlabel(i)
            plt.ylabel("frequencia")
            plt.show()


def plot_disp(df: pd.DataFrame, numeric_cols: list, target: str = None) -> None:
    """Take the numeric columns and plot a scatter plot

    Arguments:
        df: original dataset to be plot
        numeric_cols:    the list of the numerics series
    """
    for a, i in enumerate(numeric_cols):
        if target is not None:
            df.plot.scatter(i, target)
        else:
            for i2 in numeric_cols[a + 1 :]:
                df.plot.scatter(i, i2)


def plot_hexbin(df: pd.DataFrame, x: str, y: str, grid: int, color: str, bins: str) -> None:
    """Plot a hexbin plot of two variables

    Arguments:
        df: original dataset to be plotted
        x: name of the column for the x-axis
        y: name of the column for the y-axis
        grid: size of the hexagon grid
        color: colormap to be used
        bins: specification for binning, with default None
    """
    if bins == "log":
        df.plot.hexbin(x=x, y=y, gridsize=grid, cmap=color, bins="log")
    else:
        df.plot.hexbin(x=x, y=y, gridsize=grid, cmap=color)


def plot_hist_gpy(df: pd.DataFrame, x: str, y: str, bins: int) -> pd.Series:
    """Plot a stacked histogram grouped by a specified column.

    This function takes a DataFrame and two column names,
    bins the values of the first column, groups by the second column,
    and plots a stacked histogram. It also returns the value counts of the first column.

    Arguments:
        df: The original dataset to be plotted.
        x: The name of the column for binning and x-axis.
        y: The name of the column for grouping and stacking.
        bins: The number of bins to be used for binning.

    Returns:
        pd.Series: A Series containing the counts of unique values in the first column.
    """
    df["bins"] = pd.cut(df[x], bins=bins)
    df.groupby([y, "bins"]).size().unstack(0).plot(kind="bar", stacked=True)
    z = df[x].value_counts()
    return z


def plot_facetgrid(df: pd.DataFrame, x: str, hue: str, height: float, xlim: tuple) -> None:
    """Plot a FacetGrid with a KDE plot for a specified column, colored by another column.

    This function creates a seaborn FacetGrid, which is a multi-plot grid for plotting
    conditional relationships. A KDE (Kernel Density Estimate) plot for the specified
    column `x` is created, with coloring based on the unique values of column `hue`.

    Arguments:
        df: The original dataset to be plotted.
        x: The name of the column for the x-axis.
        hue: The name of the column used for coloring.
        height: Height of each facet.
        xlim: Limits for x-axis (min, max)."""
    g = sns.FacetGrid(df, hue=hue, height=height)
    g.map_dataframe(sns.kdeplot, x=x)
    g.add_legend()
    g.set(xlim=xlim)


def correlation(df: pd.DataFrame, cols_not_numeric: list, x: str) -> plot:
    """Calculate the Pearson correlation of a specified column with other columns,
    and display a styled DataFrame highlighting the correlation values.

    This function first converts specified non-numeric columns to numeric values,
    calculates the Pearson correlation, filters and prints the correlations that meet
    a specified threshold, and returns a styled DataFrame of correlations with the specified column.

    Arguments:
        df: The original dataset.
        cols_not_numeric: A list of column names to be converted to numeric.
        x: The name of the column with which to calculate the correlation.

    Returns:
        A styled DataFrame displaying the correlation values with color gradient.
    """
    for col in cols_not_numeric:
        df[col] = nf.categoric_to_numeric(df[col])
    correlation = df.corr(method="pearson")
    correlation_with_yield_pb = correlation.loc[[x]]
    dt = df.corr()[[x]].style.background_gradient(axis=0, cmap="YlOrRd")
    for i in correlation_with_yield_pb:
        if sum(correlation_with_yield_pb[i]) >= 0.2 or sum(correlation_with_yield_pb[i]) <= -0.2:
            print(correlation_with_yield_pb[i])
    return dt
