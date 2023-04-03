import numpy as np
import torch
from enum import Enum
import random


class RECOMMENDENTATION_TYPE(Enum):
    USER_TO_ITEM = "USER_TO_ITEM"
    ITEM_TO_ITEM = "ITEM_TO_ITEM"


def get_entity2dict(df, id_col):
    entity2dict = {}

    for idx, _id in enumerate(df[id_col].to_list()):
        entity2dict[_id] = idx

    return entity2dict


def get_similar_listings_by_graph_embeddings(
    query_listing_idx, listing_embeddings, reverse_test_listings2dict, K
):
    """
    Generate the top-k closest listings with the first listing the reviewers has reviewed in terms of embeddings
    """
    query_listing_embedding = listing_embeddings[query_listing_idx]
    cos_t = torch.nn.CosineSimilarity(dim=1)(query_listing_embedding, listing_embeddings)
    top_ids = torch.argsort(-cos_t).numpy()
    query_listing_id = reverse_test_listings2dict[query_listing_idx]
    recommendation_list = []
    for index in top_ids:
        if len(recommendation_list) == K:
            break
        candidate_listing_id = reverse_test_listings2dict[index]
        if candidate_listing_id != query_listing_id:
            recommendation_list.append(candidate_listing_id)
    return recommendation_list


def get_recommended_listings_by_graph_embeddings(
    query_user_embedding, listing_embeddings, reverse_test_listings2dict, K
):
    """
    for each user, generate the top-k closest listing in terms of embeddings
    """
    cos_t = torch.nn.CosineSimilarity(dim=1)(query_user_embedding, listing_embeddings)
    top_ids = torch.argsort(-cos_t).numpy()
    recommendation_list = [reverse_test_listings2dict[index] for index in top_ids[:K]]
    return recommendation_list


def prepare_evaluation_pairs(
    rec_type,
    test_reviews,
    test_listings,
    user_embeddings,
    listing_embeddings,
    test_listings2dict,
    reverse_test_listings2dict,
    test_reviewers2dict,
    reverse_test_reviewers2dict,
    K,
):
    # Generate (num_user, K) recommendation matrix
    # Generate (num_user, x) ground truth matrix where x is unsure
    recommendations = []
    ground_truths = []
    if rec_type == RECOMMENDENTATION_TYPE.USER_TO_ITEM.value:
        n_users = user_embeddings.shape[0]
        for user_idx in range(n_users):
            query_user_embedding = user_embeddings[user_idx]
            recommendation_list = get_recommended_listings_by_graph_embeddings(
                query_user_embedding, listing_embeddings, reverse_test_listings2dict, K
            )
            query_user_id = reverse_test_reviewers2dict[user_idx]
            ground_truth_list = list(
                test_reviews[test_reviews["reviewer_id"] == query_user_id]["listing_id"].values
            )
            recommendations.append(recommendation_list)
            ground_truths.append(ground_truth_list)

    elif rec_type == RECOMMENDENTATION_TYPE.ITEM_TO_ITEM.value:
        v = test_reviews["reviewer_id"].value_counts()
        reviews_that_user_interaction_more_than_once = test_reviews[
            test_reviews["reviewer_id"].isin(v.index[v.gt(1)])
        ]
        for reviewer_id in reviews_that_user_interaction_more_than_once["reviewer_id"].unique():
            user_interactions = reviews_that_user_interaction_more_than_once[
                reviews_that_user_interaction_more_than_once["reviewer_id"] == reviewer_id
            ]
            assert len(user_interactions) > 1
            user_first_interaction = user_interactions.sort_values(by="created_at").iloc[0]
            query_listing_id = user_first_interaction["listing_id"]
            query_listing_idx = test_listings2dict[query_listing_id]
            recommendation_list = get_similar_listings_by_graph_embeddings(
                query_listing_idx, listing_embeddings, reverse_test_listings2dict, K
            )
            ground_truth_list = user_interactions[
                ~user_interactions["id"].isin([user_first_interaction["id"]])
            ]["listing_id"].values

            assert len(ground_truth_list) != 0
            recommendations.append(recommendation_list)
            ground_truths.append(ground_truth_list)

    return np.array(recommendations), np.array(ground_truths)


def hit_rate(recommendations, ground_truths):
    n_users = recommendations.shape[0]
    hits = []

    for user_idx in range(n_users):
        recommendation = recommendations[user_idx]
        ground_truth = ground_truths[user_idx]
        if len(set(recommendation).intersection(ground_truth)) > 0:
            hit = 1
        else:
            hit = 0
        hits.append(hit)
    return np.array(hits).mean()


def evaluate_nn(
    user_embeddings,
    listing_embeddings,
    test_reviews,
    test_listings,
    test_listings2dict,
    reverse_test_listings2dict,
    test_reviewers2dict,
    reverse_test_reviewers2dict,
    K=10,
):
    u2i_recommendations, u2i_ground_truths = prepare_evaluation_pairs(
        RECOMMENDENTATION_TYPE.USER_TO_ITEM.value,
        test_reviews,
        test_listings,
        user_embeddings,
        listing_embeddings,
        test_listings2dict,
        reverse_test_listings2dict,
        test_reviewers2dict,
        reverse_test_reviewers2dict,
        K,
    )
    u2i_hit_rate = hit_rate(u2i_recommendations, u2i_ground_truths)
    print("u2i: ", u2i_hit_rate)

    i2i_recommendations, i2i_ground_truths = prepare_evaluation_pairs(
        RECOMMENDENTATION_TYPE.ITEM_TO_ITEM.value,
        test_reviews,
        test_listings,
        user_embeddings,
        listing_embeddings,
        test_listings2dict,
        reverse_test_listings2dict,
        test_reviewers2dict,
        reverse_test_reviewers2dict,
        K,
    )
    i2i_hit_rate = hit_rate(i2i_recommendations, i2i_ground_truths)
    print("i2i: ", i2i_hit_rate)

    return u2i_hit_rate, i2i_hit_rate


# TODOYH: Revert back
@torch.no_grad()
def evaluate_model(
    model,
    test_data,
    test_reviews,
    test_listings,
    test_listings2dict,
    reverse_test_listings2dict,
    test_reviewers2dict,
    reverse_test_reviewers2dict,
):
    model.eval()
    embeddings = model.inference(test_data.x_dict, test_data.edge_index_dict)
    user_embeddings = embeddings["user"]
    listing_embeddings = embeddings["listing"]
    return random.random()

    return evaluate_nn(
        user_embeddings,
        listing_embeddings,
        test_reviews,
        test_listings,
        test_listings2dict,
        reverse_test_listings2dict,
        test_reviewers2dict,
        reverse_test_reviewers2dict,
    )
