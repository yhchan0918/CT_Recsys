import wandb
import pandas as pd
from loguru import logger
import torch
import torch.nn.functional as F
from datetime import datetime as dt
import os
from dateutil.relativedelta import relativedelta  # type: ignore
import functools

from train_test_split import *
from evaluate import *
from constants import *
from model import *

wandb.login()


def initialize_run():
    try:
        api = wandb.Api()
        runs = api.runs(f"{ENTITY_NAME}/{PROJECT_NAME}")
        if len(runs) == 0:
            raise Exception("No runs")
        last_iteration = float("-inf")
        end_date_of_last_iteration = None

        for run in runs:
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}
            run_name = run.name
            iteration_no = int(run_name.split("_")[1])
            if iteration_no > last_iteration:
                last_iteration = iteration_no
                end_date_of_last_iteration = config["end_date"]

        current_iteration = last_iteration + 1
        current_start_date = dt.strptime(
            end_date_of_last_iteration, "%Y-%m-%d"
        ).date() + relativedelta(days=1)
    except Exception as e:
        print(e)
        current_iteration = 0
        current_start_date = dt.strptime(SIMULATION_START_DATE, "%Y-%m-%d").date()

    return current_iteration, current_start_date


def filter_test_data_by_scenario(train_reviews, test_reviews, user_col, scenario_type):
    if scenario_type == "cold_start_new_user":
        train_reviewers = list(train_reviews[user_col].unique())
        return test_reviews[~test_reviews[user_col].isin(train_reviewers)]


def get_nunique(df, col):
    return df[col].nunique()


# Driver code
def run():
    iteration, _ = initialize_run()
    start_date = dt.strptime("2021-10-24", "%Y-%m-%d").date()
    if start_date == dt.strptime(MAX_START_DATE, "%Y-%m-%d").date():
        raise Exception("Stop Simulation")
    end_date, nxt_start_date = split_date_by_period_months(start_date, TOTAL_MONTHS_PER_ITERATION)

    logger.info("Start of Retraining")
    print(iteration)
    print(start_date, end_date)
    directory = "/Users/yhchan/Downloads/FYP/data/processed"
    reviews = pd.read_parquet(f"{directory}/reviews_with_interactions.parquet")
    listings = pd.read_parquet(f"{directory}/listings_with_interactions.parquet")
    reviewers = pd.read_parquet(f"{directory}/reviewers_with_interactions.parquet")

    wandb.init(
        project=PROJECT_NAME,
        name=f"iteration_{iteration}",
        config={
            "architecture": "Unsupervised GraphSAGE",
            "iteration": iteration,
            "start_date": start_date,
            "end_date": end_date,
            "learning_rate": 0.01,
            "hidden_channels": 64,
            "train_batch_size": 1024,
            "test_batch_size": 256,
            "epochs": 50,
            "train_num_neighbours": [10, 10],
            "test_num_neighbours": [-1],  # So no sampling happens
            "train_split_period_months": TRAIN_SPLIT_PERIOD_MONTHS,
            "total_months_of_data": TOTAL_MONTHS_PER_ITERATION,
        },
    )
    wandb.define_metric("train_loss", step_metric="epoch", summary="min")
    wandb.define_metric("test_loss", step_metric="epoch", summary="min")

    # Split into train, test and test for cold start scenario
    (
        train_reviews,
        train_listings,
        train_reviewers,
        test_reviews,
        test_listings,
        test_reviewers,
    ) = main_train_val_test(
        reviews,
        listings,
        reviewers,
        start_date,
        end_date,
        wandb.config["train_split_period_months"],
    )
    cold_start_test_reviews = filter_test_data_by_scenario(
        train_reviews, test_reviews, "reviewer_id", "cold_start_new_user"
    )
    cold_start_test_listings, cold_start_test_reviewers = build_partitioned_data(
        cold_start_test_reviews, listings, reviewers
    )

    # Build idx to id dict and reverse version of it
    test_listings2dict = get_entity2dict(test_listings, "listing_id")
    reverse_test_listings2dict = {k: v for v, k in test_listings2dict.items()}
    test_reviewers2dict = get_entity2dict(test_reviewers, "reviewer_id")
    reverse_test_reviewers2dict = {k: v for v, k in test_reviewers2dict.items()}
    cold_start_test_listings2dict = get_entity2dict(cold_start_test_listings, "listing_id")
    reverse_cold_start_test_listings2dict = {k: v for v, k in cold_start_test_listings2dict.items()}
    cold_start_test_reviewers2dict = get_entity2dict(cold_start_test_reviewers, "reviewer_id")
    reverse_cold_start_test_reviewers2dict = {
        k: v for v, k in cold_start_test_reviewers2dict.items()
    }

    # Build Graph
    data = build_heterograph(reviews, listings)
    train_data = build_heterograph(train_reviews, train_listings)
    test_data = build_heterograph(test_reviews, test_listings)
    cold_start_test_data = build_heterograph(cold_start_test_reviews, cold_start_test_listings)
    print("Training Heterogenous Graph", train_data)
    print("Test Heterogenous Graph", test_data)
    print("Test Heterogenous Graph (Cold Start Scenerio)", cold_start_test_data)

    train_loader = prepare_data_loader(
        data=train_data,
        batch_size=wandb.config["train_batch_size"],
        num_neighbours=wandb.config["train_num_neighbours"],
    )
    test_loader = prepare_data_loader(
        data=test_data,
        batch_size=wandb.config["test_batch_size"],
        num_neighbours=wandb.config["test_num_neighbours"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)

    train_test_data_dict = {
        "num_reviews": len(train_reviews) + len(test_reviews),
        "num_train_reviews": len(train_reviews),
        "num_test_reviews": len(test_reviews),
        "num_cold_start_test_reviews": len(cold_start_test_reviews),
        "num_unique_cold_start_test_listings": get_nunique(cold_start_test_reviews, "listing_id"),
        "num_unique_cold_start_test_reviewers": get_nunique(cold_start_test_reviews, "reviewer_id"),
        "num_unique_train_listings": get_nunique(train_listings, "listing_id"),
        "num_unique_test_listings": get_nunique(test_listings, "listing_id"),
        "num_unique_train_reviewers": get_nunique(train_reviewers, "reviewer_id"),
        "num_unique_test_reviewers": get_nunique(test_reviewers, "reviewer_id"),
    }
    train_reviews.to_parquet("train/train_reviews.parquet", index=False)
    train_listings.to_parquet("train/train_listings.parquet", index=False)
    train_reviewers.to_parquet("train/train_reviewers.parquet", index=False)
    test_reviews.to_parquet("test/test_reviews.parquet", index=False)
    test_listings.to_parquet("test/test_listings.parquet", index=False)
    test_reviewers.to_parquet("test/test_reviewers.parquet", index=False)
    cold_start_test_reviews.to_parquet("test/cold_start_test_reviews.parquet", index=False)
    cold_start_test_listings.to_parquet("test/cold_start_test_listings.parquet", index=False)
    cold_start_test_reviewers.to_parquet("test/cold_start_test_reviewers.parquet", index=False)

    print(train_test_data_dict)
    wandb.log(train_test_data_dict)
    dataset_art = wandb.Artifact(f"{start_date}_{end_date}_data_{iteration}", type="dataset")
    for dir in ["train", "test"]:
        dataset_art.add_dir(dir)
    wandb.log_artifact(dataset_art)

    # Modelling
    model = Model(hidden_channels=wandb.config["hidden_channels"], data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])

    def train():
        model.train(True)
        total_loss = 0
        # Why using mini-batch gradient descent
        # Update NN multiple times every epoch, Make more precise update to the parameters by calculating the average loss in each step
        # Reduce overall training time and num of required epochs for reaching convergence, computational efficiency
        for batch in train_loader:
            batch = batch.to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()
            # Make predictions for this batch
            h = model(batch.x_dict, batch.edge_index_dict)
            h_src = h["user"][batch["user", "listing"].edge_label_index[0]]
            h_dst = h["listing"][batch["user", "listing"].edge_label_index[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            # Compute the loss and its gradients
            loss = F.binary_cross_entropy_with_logits(pred, batch["user", "listing"].edge_label)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            total_loss += float(loss) * pred.size(0)

        train_loss = total_loss / train_data.num_nodes
        return train_loss

    @torch.no_grad()
    def test(test_data_loader, test_data, model):
        model.eval()
        total_loss = 0
        for batch in test_data_loader:
            batch = batch.to(device)
            # Make predictions for this batch
            h = model(batch.x_dict, batch.edge_index_dict)
            h_src = h["user"][batch["user", "listing"].edge_label_index[0]]
            h_dst = h["listing"][batch["user", "listing"].edge_label_index[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            # Compute the loss and its gradients
            loss = F.binary_cross_entropy_with_logits(pred, batch["user", "listing"].edge_label)
            total_loss += float(loss) * pred.size(0)

        test_loss = total_loss / test_data.num_nodes
        return test_loss

    best_train_loss = float("inf")
    best_test_loss = float("inf")
    best_model_path = None
    # Train and Evaluate Loss
    test_wrapper = functools.partial(test, test_loader, test_data)
    for epoch in range(1, wandb.config["epochs"] + 1):
        model_is_best = False
        train_loss = train()
        test_loss = test_wrapper(model)

        if train_loss < best_train_loss:
            wandb.run.summary["best_train_loss"] = train_loss
            best_train_loss = train_loss

        if test_loss < best_test_loss:
            wandb.run.summary["best_test_loss"] = test_loss
            best_test_loss = test_loss
            model_is_best = True

        metrics_dict = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch": epoch,
        }
        wandb.log(metrics_dict)
        logger.info(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} "
        )

        model_path = f"./models/{epoch}_model_state_dict.pt"
        torch.save(model.state_dict(), model_path)
        model_art = wandb.Artifact(f"{MODEL_NAME}_iteration_{iteration}", type="model")
        model_art.add_file(model_path)
        wandb.log_artifact(
            model_art,
            aliases=[
                LATEST_TAG,
            ]
            if model_is_best
            else None,
        )
        if model_is_best:
            best_model_path = model_path

    # Query production model
    production_model = None
    try:
        production_model_dir = wandb.use_artifact(
            f"{ENTITY_NAME}/{PROJECT_NAME}/{MODEL_REGISTRY_NAME}:{PRODUCTION_TAG}"
        ).download()
        production_model_path = f"{production_model_dir}/{os.listdir(production_model_dir)[0]}"
        production_model = load_model(production_model_path, wandb.config["hidden_channels"], data)
        logger.info("Production model found")
    except Exception as e:
        print(e)
        logger.info("No production model found")

    # Evaluation
    evaluate_model_wrapper = functools.partial(
        evaluate_model,
        test_data,
        test_reviews,
        test_listings2dict,
        reverse_test_listings2dict,
        reverse_test_reviewers2dict,
    )
    evaluate_model_in_cold_start_wrapper = functools.partial(
        evaluate_model,
        cold_start_test_data,
        cold_start_test_reviews,
        cold_start_test_listings2dict,
        reverse_cold_start_test_listings2dict,
        reverse_cold_start_test_reviewers2dict,
    )
    contender_model = load_model(best_model_path, wandb.config["hidden_channels"], data)
    contender_model_u2i_hit_rate, contender_model_i2i_hit_rate = evaluate_model_wrapper(
        contender_model,
    )
    (
        contender_model_u2i_hit_rate_in_cold_start,
        contender_model_i2i_hit_rate_in_cold_start,
    ) = evaluate_model_in_cold_start_wrapper(
        contender_model,
    )
    wandb.log(
        {
            "contender_model_u2i_hit_rate": contender_model_u2i_hit_rate,
            "contender_model_i2i_hit_rate": contender_model_i2i_hit_rate,
            "contender_model_u2i_hit_rate_in_cold_start": contender_model_u2i_hit_rate_in_cold_start,
            "contender_model_i2i_hit_rate_in_cold_start": contender_model_i2i_hit_rate_in_cold_start,
        }
    )

    if production_model == None:
        is_contender_model_better = True
    else:
        test_loss_by_production_model = test_wrapper(production_model)
        production_model_u2i_hit_rate, production_model_i2i_hit_rate = evaluate_model_wrapper(
            production_model,
        )
        (
            production_model_u2i_hit_rate_in_cold_start,
            production_model_i2i_hit_rate_in_cold_start,
        ) = evaluate_model_in_cold_start_wrapper(
            production_model,
        )
        wandb.log(
            {
                "production_model_test_loss": test_loss_by_production_model,
                "production_model_u2i_hit_rate": production_model_u2i_hit_rate,
                "production_model_i2i_hit_rate": production_model_i2i_hit_rate,
                "production_model_u2i_hit_rate_in_cold_start": production_model_u2i_hit_rate_in_cold_start,
                "production_model_i2i_hit_rate_in_cold_start": production_model_i2i_hit_rate_in_cold_start,
            }
        )
        is_contender_model_better = test_loss_by_production_model < best_test_loss and (
            contender_model_u2i_hit_rate > production_model_u2i_hit_rate
            or contender_model_i2i_hit_rate > production_model_i2i_hit_rate
        )

    model_registry_art = wandb.Artifact(MODEL_REGISTRY_NAME, type="model_registry")
    model_registry_art.add_file(best_model_path)
    # By right, it should go through shadow deployment or A/B Testing first before promoting it to production
    wandb.log_artifact(
        model_registry_art,
        aliases=[PRODUCTION_TAG, LATEST_TAG] if is_contender_model_better else [LATEST_TAG],
    )
    # Save code
    wandb.run.log_code(
        "./", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb")
    )
    wandb.finish()
    logger.info("End of Retraining")


run()
