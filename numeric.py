from sklearn.preprocessing import LabelEncoder
import pandas as pd


class NumericFunctions:
    @classmethod
    def to_numeric(cls, col: pd.Series) -> pd.Series:
        """Receives a Col and transforms it into a numeric

        Arguments:
            col: Series of values that it will be cleaned

        Returns:
            col: Clean numeric columns
        """
        col = col.replace("[^0-9,]", "", regex=True)
        col = col.replace(",", ".", regex=True)
        col = col.replace("^$", "-1", regex=True)
        col = col.astype(float)
        col = col.replace(-1, "", regex=True)
        return col

    @classmethod
    def categoric_to_numeric(cls, col: pd.Series) -> pd.Series:
        """Receives a Col and transforms it into a numeric

        Arguments:
            col: Series of values that it be transform

        Returns:
            col: Categoric - numeric columns
        """
        col = LabelEncoder().fit_transform(col)
        return col
