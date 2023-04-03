import json
import os
import pandas as pd
import numpy as np

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=3)

reviews = pd.read_parquet("reviews.parquet")
reviewers = pd.DataFrame(dict(reviewer_id=reviews["reviewer_id"].unique()))
reviewers_cols = ["reviewer_picture_url"]


def get_val_by_reviewer_id(reviewer_id, col):
    return reviews[reviews["reviewer_id"] == reviewer_id].iloc[0][col]


for col in reviewers_cols:
    reviewers[col] = reviewers["reviewer_id"].parallel_apply(
        lambda reviewer_id: get_val_by_reviewer_id(reviewer_id, col)
    )
reviewers.to_parquet("reviewers.parquet", index=False)
