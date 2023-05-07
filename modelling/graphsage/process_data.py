# process_data.py
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import pandas as pd
import datetime
import pandas as pd
from typing import Tuple
from dateutil.relativedelta import relativedelta  # type: ignore
from loguru import logger

from constants import REVIEWS_DATE_COL, LISTING_COLS, COMMENT_EMBEDDING_DIMENSION


def split_df_date(
    df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame:
    """
    Filter the data from specified start date to specified end date
    """
    assert REVIEWS_DATE_COL in df.columns.tolist(), f"Missing {REVIEWS_DATE_COL}, cannot split df"
    date_column = df[REVIEWS_DATE_COL].dt.date
    mask = (date_column >= start_date) & (date_column <= end_date)
    df_split = df.loc[mask]
    return df_split


def check_split_period_months(
    end_date: datetime.date,
    end_date_train: datetime.date,
) -> None:
    """
    Validate the amount of split_period_months is not greater than the period of the rest of the data
    """
    if end_date_train >= end_date:
        raise Exception(
            "train_split_period_months is greater than the period of the entire dataset"
        )


def split_date_by_period_months(
    current_split_start_date: datetime.date, period_months: int
) -> Tuple[datetime.date, datetime.date]:
    """
    Add up the period_months to the start date
    to get the end date of the current split and
    start date of the next split
    """
    current_split_end_date = current_split_start_date + relativedelta(months=period_months, days=-1)
    next_split_start_date = current_split_end_date + relativedelta(days=1)

    return current_split_end_date, next_split_start_date


def get_split_dates(
    df: pd.DataFrame,
    start_date: datetime.datetime,
    train_split_period_months: int,
) -> Tuple[
    datetime.date,
    datetime.date,
    datetime.date,
    datetime.date,
    datetime.date,
    datetime.date,
]:
    """
    Get the dates used to split data into train and test datasets
    """
    start_date_ = start_date
    end_date = df[REVIEWS_DATE_COL].dt.date.max()
    end_date_train, start_date_test = split_date_by_period_months(
        start_date_, train_split_period_months
    )
    check_split_period_months(end_date, end_date_train)

    return (
        start_date_,
        end_date_train,
        start_date_test,
        end_date,
    )


def set_train_test(
    df: pd.DataFrame,
    start_date: datetime.date,
    end_date_train: datetime.date,
    start_date_test: datetime.date,
    end_date: datetime.date,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the datasets into train, valid and test datatsets
    """
    try:
        train_df = split_df_date(
            df,
            start_date=start_date,
            end_date=end_date_train,
        )
        test_df = split_df_date(
            df,
            start_date=start_date_test,
            end_date=end_date,
        )
        return train_df, test_df
    except AssertionError:
        message = "Error splitting data into train and test"
        logger.exception(message)
        raise Exception(message)

def retrieve_unique_reviewers(reviews):
    # Use the latest comment of each user to represent the user profile
    reviewers = reviews.sort_values(by=REVIEWS_DATE_COL).drop_duplicates(
        subset=["reviewer_id"], keep="last"
    )
    return reviewers

def main_train_test(
    reviews, listings, start_date, end_date, train_split_period_months
):
    df = split_df_date(reviews, start_date, end_date)
    (
        start_date,
        end_date_train,
        start_date_test,
        end_date,
    ) = get_split_dates(df, start_date, train_split_period_months)
    train_reviews, test_reviews = set_train_test(
        df,
        start_date,
        end_date_train,
        start_date_test,
        end_date,
    )

    train_listings, train_reviewers = build_partitioned_data(train_reviews, listings)
    test_listings, test_reviewers = build_partitioned_data(test_reviews, listings)
    logger.info("Split df into train and test portion")

    return (
        train_reviews,
        train_listings,
        train_reviewers,
        test_reviews,
        test_listings,
        test_reviewers,
    )


def build_partitioned_data(reviews, listings):
    _listings = listings.copy()
    _reviews = reviews.copy()
    distinct_listings_in_reviews = _reviews["listing_id"].unique()
    _listings = _listings[_listings["listing_id"].isin(distinct_listings_in_reviews)]
    _reviewers = retrieve_unique_reviewers(_reviews)
    return _listings, _reviewers


def load_node_from_df(df, index_col, features_cols=None):
    mapping = {index: i for i, index in enumerate(df[index_col].unique())}

    x = None
    if features_cols:
        xs = []
        for col in features_cols:
            if df[col].dtype == float:
                val = df[col].values
            else:
                val = df[col].cat.codes.values
            temp = torch.from_numpy(val).view(-1, 1).to(torch.float32)
            xs.append(temp)
        x = torch.cat(xs, dim=-1).to(torch.float32)
    return x, mapping


def load_edge_from_df(df, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst]).to(torch.long)

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.flatten(torch.cat(edge_attrs, dim=-1))

    return edge_index, edge_attr


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


def build_heterograph(reviews, listings, reviewers, include_rating_as_edge_label=False):

    listing_features_cols = LISTING_COLS["features_cols"]
    # Convert data type from boolean to categorical
    for col in listings[listing_features_cols].select_dtypes(include=["bool"]):
        listings[col] = listings[col].astype("category")

    user_features_cols = [f"comment_embedding_{i}" for i in range(COMMENT_EMBEDDING_DIMENSION)]
    user_x, user_mapping = load_node_from_df(
        reviewers, index_col="reviewer_id", features_cols=user_features_cols
    )
    listing_x, listing_mapping = load_node_from_df(
        listings, index_col="listing_id", features_cols=listing_features_cols
    )
    edge_index, edge_label = load_edge_from_df(
        reviews,
        src_index_col="reviewer_id",
        src_mapping=user_mapping,
        dst_index_col="listing_id",
        dst_mapping=listing_mapping,
        encoders={"rating": IdentityEncoder(dtype=torch.long)},
    )

    data = HeteroData()
    data["listing"].x = listing_x
    data["user"].x = user_x
    data["user", "rates", "listing"].edge_index = edge_index
    if include_rating_as_edge_label:
        data["user", "rates", "listing"].edge_label = edge_label
        data["user", "rates", "listing"].edge_label_index = edge_index
    # We can now convert `data` into an appropriate format for training a
    # graph-based machine learning model:

    # 1. Add a reverse ('listing', 'rev_rates', 'user') relation for message passing.
    data = ToUndirected()(data)
    del data["listing", "rev_rates", "user"].edge_label  # Remove "reverse" label.
    return data

