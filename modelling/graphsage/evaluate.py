import numpy as np
from sklearn.cluster import KMeans
import torch
from enum import Enum
import pandas as pd
from collections import Counter


class RECOMMENDENTATION_TYPE(Enum):
    USER_TO_ITEM = "USER_TO_ITEM"
    ITEM_TO_ITEM = "ITEM_TO_ITEM"


def get_entity2dict(df, id_col):
    entity2dict = {}

    for idx, _id in enumerate(df[id_col].to_list()):
        entity2dict[_id] = idx

    return entity2dict


def get_similar_listings_by_graph_embeddings(
    query_listing_idx, listing_embeddings, reverse_listings2dict, K
):
    """
    Generate the top-k closest listings with the first listing the reviewers has reviewed in terms of embeddings
    """
    query_listing_embedding = listing_embeddings[query_listing_idx]
    cos_t = torch.nn.CosineSimilarity(dim=1)(query_listing_embedding, listing_embeddings)
    top_ids = torch.argsort(-cos_t).numpy()
    query_listing_id = reverse_listings2dict[query_listing_idx]
    recommendation_list = []
    for index in top_ids:
        if len(recommendation_list) == K:
            break
        candidate_listing_id = reverse_listings2dict[index]
        if candidate_listing_id != query_listing_id:
            recommendation_list.append(candidate_listing_id)
    return recommendation_list


def get_recommended_listings_by_graph_embeddings(
    query_user_embedding, listing_embeddings, reverse_listings2dict, K
):
    """
    for each user, generate the top-k closest listing in terms of embeddings
    """
    cos_t = torch.nn.CosineSimilarity(dim=1)(query_user_embedding, listing_embeddings)
    top_ids = torch.argsort(-cos_t).numpy()
    recommendation_list = [reverse_listings2dict[index] for index in top_ids[:K]]
    return recommendation_list


def prepare_evaluation_pairs(
    rec_type,
    user_dict,
    listing_embeddings,
    listings2dict,
    reverse_listings2dict,
    test_reviews,
    test_reviewers,
    K,
):
    # Generate (num_user, K) recommendation matrix
    # Generate (num_user, x) ground truth matrix where x is dependent on the available interactions in test set
    recommendations = []
    ground_truths = []
    if rec_type == RECOMMENDENTATION_TYPE.USER_TO_ITEM.value:
        for _, row in test_reviewers.iterrows():
            query_user_id = row["reviewer_id"]
            query_user_embedding = user_dict[query_user_id]
            recommendation_list = get_recommended_listings_by_graph_embeddings(
                query_user_embedding, listing_embeddings, reverse_listings2dict, K
            )
            ground_truth_list = list(
                test_reviews[test_reviews["reviewer_id"] == query_user_id]["listing_id"].values
            )
            recommendations.append(recommendation_list)
            ground_truths.append(ground_truth_list)

    elif rec_type == RECOMMENDENTATION_TYPE.ITEM_TO_ITEM.value:
        v = test_reviews["reviewer_id"].value_counts()
        # Take users who have more than one interactions so that the first interaction can be treated as
        # query listing while the later interactions could be designated as ground truth
        reviews_whose_reviewers_interaction_more_than_once = test_reviews[
            test_reviews["reviewer_id"].isin(v.index[v.gt(1)])
        ]
        for reviewer_id in reviews_whose_reviewers_interaction_more_than_once[
            "reviewer_id"
        ].unique():
            user_interactions = reviews_whose_reviewers_interaction_more_than_once[
                reviews_whose_reviewers_interaction_more_than_once["reviewer_id"] == reviewer_id
            ]
            user_first_interaction = user_interactions.sort_values(by="created_at").iloc[0]
            query_listing_id = user_first_interaction["listing_id"]
            query_listing_idx = listings2dict[query_listing_id]
            recommendation_list = get_similar_listings_by_graph_embeddings(
                query_listing_idx, listing_embeddings, reverse_listings2dict, K
            )
            ground_truth_list = user_interactions[
                ~user_interactions["id"].isin([user_first_interaction["id"]])
            ]["listing_id"].values

            recommendations.append(recommendation_list)
            ground_truths.append(ground_truth_list)

    return np.array(recommendations), np.array(ground_truths)


def hit_rate(recommendations, ground_truths):
    n_users = recommendations.shape[0]
    hits = []
    print(
        f"Length of recommendations: {len(recommendations)}, Length of ground truth: {len(ground_truths)}",
    )
    for user_idx in range(n_users):
        recommendation = recommendations[user_idx]
        ground_truth = ground_truths[user_idx]
        if len(set(recommendation).intersection(ground_truth)) > 0:
            hit = 1
        else:
            hit = 0
        hits.append(hit)

    print("Count of hit & non-hit: ", Counter(hits))
    return np.array(hits).mean()


def coverage(recommendations, available_listings):
    unique_recommendations = np.unique(np.array(recommendations))
    n_unique_recommendations = len(unique_recommendations)
    n_available_listings = len(available_listings)

    print("n_unique_recommendations: ", n_unique_recommendations)
    print("n_available_listings: ", n_available_listings)
    return n_unique_recommendations / n_available_listings


def eval_recsys(
    user_dict,
    listing_embeddings,
    listings2dict,
    reverse_listings2dict,
    test_reviews,
    test_reviewers,
    K,
):
    available_listings = list(listings2dict.keys())
    print("Start u2i recommendation")
    u2i_recommendations, u2i_ground_truths = prepare_evaluation_pairs(
        RECOMMENDENTATION_TYPE.USER_TO_ITEM.value,
        user_dict,
        listing_embeddings,
        listings2dict,
        reverse_listings2dict,
        test_reviews,
        test_reviewers,
        K,
    )
    u2i_hit_rate = hit_rate(u2i_recommendations, u2i_ground_truths)
    u2i_coverage = coverage(u2i_recommendations, available_listings)
    print("u2i hit rate: ", u2i_hit_rate)
    print("u2i coverage: ", u2i_coverage)

    print("Start i2i recommendation")
    i2i_recommendations, i2i_ground_truths = prepare_evaluation_pairs(
        RECOMMENDENTATION_TYPE.ITEM_TO_ITEM.value,
        user_dict,
        listing_embeddings,
        listings2dict,
        reverse_listings2dict,
        test_reviews,
        test_reviewers,
        K,
    )
    i2i_hit_rate = hit_rate(i2i_recommendations, i2i_ground_truths)
    i2i_coverage = coverage(i2i_recommendations, available_listings)
    print("i2i hit rate: ", i2i_hit_rate)
    print("i2i coverage: ", i2i_coverage)

    return u2i_hit_rate, u2i_coverage, i2i_hit_rate, i2i_coverage


@torch.no_grad()
def generate_user_and_listing_embeddings(graph_data, users, model):
    print("Start generating embeddings")

    model.eval()
    embeddings = model.inference(graph_data.x_dict, graph_data.edge_index_dict)
    print("Finish generating embeddings")
    user_embeddings = embeddings["user"]
    listing_embeddings = embeddings["listing"]

    user_dict = {}

    assert len(user_embeddings) == len(users)

    user_idx = 0
    for _, row in users.iterrows():
        user_id = row["reviewer_id"]
        user_dict[user_id] = user_embeddings[user_idx]
        user_idx += 1

    return user_dict, listing_embeddings


def run_eval(
    involved_data,
    involved_reviewers,
    involved_listings2dict,
    reverse_involved_listings2dict,
    test_reviews,
    test_reviewers,
    K,
    model,
):
    print(f"K = {K}")
    # Generate embedding for all users and listings
    user_dict, listing_embeddings = generate_user_and_listing_embeddings(
        involved_data, involved_reviewers, model
    )

    # Evaluate the performance of recsys by recommending top K similar listings to user in terms of embedding similarity

    return eval_recsys(
        user_dict,
        listing_embeddings,
        involved_listings2dict,
        reverse_involved_listings2dict,
        test_reviews,
        test_reviewers,
        K,
    )


def generate_user_and_listing_embeddings_with_semi_personalization(
    n_clusters, involved_data, involved_reviewers, cold_start_test_reviewers, model
):
    # Generate embedding for all users and listings
    user_dict, listing_embeddings = generate_user_and_listing_embeddings(
        involved_data, involved_reviewers, model
    )

    # Create a list of dictionaries for user embeddings
    user_list = [
        {"user_id": user_id, "embedding": embedding.numpy()}
        for user_id, embedding in user_dict.items()
    ]

    # Perform clustering on all users embeddings
    # Initialize k-means algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++")
    print("Running K-Means Algorithm...")
    # Get the cluster assignments for each input embedding
    cluster_labels = kmeans.fit_predict(
        np.array([user_item["embedding"] for user_item in user_list])
    )
    print("Complete running K-Means Algorithm...")

    # Get the centroid for each cluster
    cluster_centroids = kmeans.cluster_centers_

    # Replace cold start user embedding with centroid embedding
    cold_start_test_reviewer_ids = cold_start_test_reviewers["reviewer_id"].unique()
    cold_start_test_reviewer_indices = np.where(
        np.isin(np.array([user["user_id"] for user in user_list]), cold_start_test_reviewer_ids)
    )[0]
    print("cold_start_test_reviewer_indices", cold_start_test_reviewer_indices)
    for i in cold_start_test_reviewer_indices:
        print("i", i)
        cluster_label = cluster_labels[i]
        user_dict[user_list[i]["user_id"]] = torch.from_numpy(cluster_centroids[cluster_label])

    return user_dict, listing_embeddings


def run_eval_with_semi_personalization(
    involved_data,
    involved_reviewers,
    involved_listings2dict,
    reverse_involved_listings2dict,
    cold_start_test_reviews,
    cold_start_test_reviewers,
    K,
    n_clusters,
    model,
):

    (
        user_dict_with_semi_personalization,
        listing_embeddings_with_semi_personalization,
    ) = generate_user_and_listing_embeddings_with_semi_personalization(
        n_clusters, involved_data, involved_reviewers, cold_start_test_reviewers, model
    )

    return eval_recsys(
        user_dict_with_semi_personalization,
        listing_embeddings_with_semi_personalization,
        involved_listings2dict,
        reverse_involved_listings2dict,
        cold_start_test_reviews,
        cold_start_test_reviewers,
        K,
    )


def filter_test_data_by_scenario(train_reviews, test_reviews, user_col, scenario_type):
    if scenario_type == "cold_start_new_user":
        train_reviewers = list(train_reviews[user_col].unique())
        return test_reviews[~test_reviews[user_col].isin(train_reviewers)]
