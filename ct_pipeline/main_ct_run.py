import wandb
import pandas as pd
from loguru import logger
import torch
import torch.nn.functional as F
from datetime import datetime as dt
import os
from dateutil.relativedelta import relativedelta  # type: ignore

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


# Driver code
def run():
    iteration, start_date = initialize_run()
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
            "hidden_channels": 32,
            "train_batch_size": 1024,
            "test_batch_size": 256,
            "epochs": 2,  # TODOYH
            "train_num_neighbours": [10, 10],
            "test_num_neighbours": [-1],
            "train_split_period_months": 10,
            "total_months_of_data": TOTAL_MONTHS_PER_ITERATION,
        },
    )
    wandb.define_metric("train_loss", step_metric="epoch", summary="min")
    wandb.define_metric("test_loss", step_metric="epoch", summary="min")

    # Prepare data and graph
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
    wandb.log(
        {
            "num_reviews": len(train_reviews) + len(test_reviews),
            "num_train_reviews": len(train_reviews),
            "num_test_reviews": len(test_reviews),
            "num_listings": len(train_listings) + len(test_listings),
            "num_train_listings": len(train_listings),
            "num_test_listings": len(test_listings),
            "num_reviewers": len(train_reviewers) + len(test_reviewers),
            "num_train_reviewers": len(train_reviewers),
            "num_test_reviewers": len(test_reviewers),
        }
    )
    train_reviews.to_csv("train/train_reviews.parquet", index=False)
    train_listings.to_csv("train/train_listings.parquet", index=False)
    train_reviewers.to_csv("train/train_reviewers.parquet", index=False)
    test_reviews.to_csv("test/test_reviews.parquet", index=False)
    test_listings.to_csv("test/test_listings.parquet", index=False)
    test_reviewers.to_csv("test/test_reviewers.parquet", index=False)

    dataset_art = wandb.Artifact(f"{start_date}_{end_date}_data_{iteration}", type="dataset")
    for dir in ["train", "test"]:
        dataset_art.add_dir(dir)
    wandb.log_artifact(dataset_art)
    test_listings2dict = get_entity2dict(test_listings, "listing_id")
    reverse_test_listings2dict = {k: v for v, k in test_listings2dict.items()}
    test_reviewers2dict = get_entity2dict(test_reviewers, "reviewer_id")
    reverse_test_reviewers2dict = {k: v for v, k in test_reviewers2dict.items()}
    data = build_heterograph(reviews, listings)
    train_data = build_heterograph(train_reviews, train_listings)
    test_data = build_heterograph(test_reviews, test_listings)
    print("Training Heterogenous Graph", train_data)
    print("Test Heterogenous Graph", test_data)

    # Modelling
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
    def test(model, test_data_loader, test_data):
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
    for epoch in range(1, wandb.config["epochs"] + 1):
        model_is_best = False
        train_loss = train()
        test_loss = test(model, test_loader, test_data)

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

    contender_model = load_model(best_model_path, wandb.config["hidden_channels"], data)
    hit_rate_by_contender_model = evaluate_model(
        contender_model,
        test_data,
        test_reviews,
        test_listings,
        test_listings2dict,
        reverse_test_listings2dict,
        test_reviewers2dict,
        reverse_test_reviewers2dict,
    )
    wandb.log({"contender_model_hit_rate": hit_rate_by_contender_model})

    if production_model == None:
        is_contender_model_better = True
    else:
        test_loss_by_production_model = test(production_model, test_loader, test_data)
        hit_rate_by_production_model = evaluate_model(
            production_model,
            test_data,
            test_reviews,
            test_listings,
            test_listings2dict,
            reverse_test_listings2dict,
            test_reviewers2dict,
            reverse_test_reviewers2dict,
        )
        wandb.log(
            {
                "production_model_test_loss": test_loss_by_production_model,
                "production_model_hit_rate": hit_rate_by_contender_model,
            }
        )
        is_contender_model_better = (
            test_loss_by_production_model < best_test_loss
            and hit_rate_by_contender_model > hit_rate_by_production_model
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
