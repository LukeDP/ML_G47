import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import os


def nulls_values_by_column(df, plot_size=(12, 8)):
    null_counts = df.isnull().sum()
    plot = null_counts.plot(kind="bar", title="Number of null values per column", figsize = plot_size)
    for i, v in enumerate(null_counts):
        plot.text(i, v, str(v), ha="center", va="bottom")
    return plot



def plot_unique_counts(counts_df, threshold, plot_size):
    ax = counts_df.plot(kind="bar", figsize=plot_size)
    for bar in ax.patches:
        color = "green" if bar.get_height() > threshold else "red"
        bar.set_color(color)
    plt.axhline(y=threshold, color="k", linestyle="--")
    plt.show()


def unique_values_by_column(df, threshold=0, plot_size=(12, 8)):
    counts = {col: df[col].nunique() for col in df.columns}
    counts_df = pd.DataFrame.from_dict(counts, orient="index", columns=["count"])
    plot_unique_counts(counts_df, threshold, plot_size)


def filter_column_uniques(df, size=1):
    df_clean = df.copy()
    for col in df.columns:
        if len(pd.unique(df[col])) <= size:
            df_clean.drop(col, axis=1, inplace=True)
    return df_clean


def histogram_plot(df, max_zscore=3, plot_size=(12, 8)):
    numerical_columns = df.select_dtypes(include=["number"]).columns

    num_columns = len(numerical_columns)
    num_rows = (num_columns + 1) // 2

    fig, axes = plt.subplots(num_rows, 2, figsize=plot_size)

    for i, column in enumerate(numerical_columns):
        row = i // 2
        col = i % 2

        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        std_dev = df[column].std()
        ax.axvline(
            x=df[column].mean() - std_dev, color="g", linestyle="--", label="std dev"
        )
        ax.axvline(x=df[column].mean() + std_dev, color="g", linestyle="--")
        ax.axvline(
            x=df[column].mean() - std_dev * max_zscore,
            color="r",
            linestyle="--",
            label="z-score",
        )
        ax.axvline(
            x=df[column].mean() + std_dev * max_zscore, color="r", linestyle="--"
        )
        ax.fill_betweenx(
            ax.get_ylim(),
            df[column].mean() - std_dev * max_zscore,
            df[column].mean() + std_dev * max_zscore,
            alpha=0.1,
            color="g",
        )
        ax.legend()

    if num_columns % 2 != 0:
        fig.delaxes(axes[num_rows - 1, 1])

    plt.tight_layout()
    return plt


def scatter_plot(df, plot_size=(15, 10)):
    plt.figure(figsize=plot_size)
    sns.pairplot(df.select_dtypes(include=["number"]), diag_kind="kde", plot_kws={"alpha": 0.5})
    plt.show()


def filter_by_zscore(df, threshold=3, exclude=[]):
    eligible_columns = df.select_dtypes(include=["int", "float"]).columns
    eligible_columns = list(set(eligible_columns) - set(exclude))
    z_scores = np.abs(stats.zscore(df[eligible_columns]))
    return set(np.where(z_scores > threshold)[0])