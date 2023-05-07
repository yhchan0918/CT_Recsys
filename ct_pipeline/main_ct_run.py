import wandb
import pandas as pd
from loguru import logger
import torch
from datetime import datetime as dt
import os
from dateutil.relativedelta import relativedelta  # type: ignore
import functools

from process_data import *
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

    config = {
        "iteration":iteration,
        "architecture": "Rating-Weighted GraphSAGE",
        "start_date": start_date,
        "end_date": end_date,
        "learning_rate": 0.01,
        "hidden_channels": 64,
        "train_batch_size": 128,
        "test_batch_size": 128,
        "epochs": 300,
        "train_num_neighbours": [10, 10],
        "test_num_neighbours": [-1],
        "train_split_period_months": 10,
        "total_months_of_data": TOTAL_MONTHS_PER_ITERATION,
        "rec_K": 5,
    }

    wandb.init(
        project=PROJECT_NAME,
        config=config,
        name=f"iteration_{iteration}"
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
    ) = main_train_test(
        reviews,
        listings,
        start_date,
        end_date,
        wandb.config["train_split_period_months"],
    )

    cold_start_test_reviews = filter_test_data_by_scenario(
        train_reviews, test_reviews, "reviewer_id", "cold_start_new_user"
    )
    cold_start_test_listings, cold_start_test_reviewers = build_partitioned_data(
        cold_start_test_reviews, listings
    )
    # Build Graph
    involved_reviews = pd.concat([train_reviews, test_reviews])
    involved_listings, involved_reviewers = build_partitioned_data(involved_reviews, listings)
    involved_data = build_heterograph(involved_reviews, involved_listings, involved_reviewers, True)
    train_data = build_heterograph(train_reviews, train_listings, train_reviewers, True)
    test_data = build_heterograph(test_reviews, test_listings, test_reviewers, True)
    cold_start_test_data = build_heterograph(
        cold_start_test_reviews, cold_start_test_listings, cold_start_test_reviewers, True
    )
    print("Whole Graph", involved_data)
    print("Training Heterogenous Graph", train_data)
    print("Test Heterogenous Graph", test_data)
    print("Test Heterogenous Graph (Cold Start Scenerio)", cold_start_test_data)

    involved_listings2dict = get_entity2dict(involved_listings, "listing_id")
    reverse_involved_listings2dict = {k: v for v, k in involved_listings2dict.items()}

    metadata_dict = {
        "num_reviews": len(involved_reviews),
        "num_train_reviews": len(train_reviews),
        "num_test_reviews": len(test_reviews),
        "num_cold_start_test_reviews": len(cold_start_test_reviews),
        "num_unique_listings": len(involved_listings),
        "num_unique_train_listings": len(train_listings),
        "num_unique_test_listings": len(test_listings),
        "num_unique_cold_start_test_listings": len(cold_start_test_listings),
        "num_unique_reviewers": len(involved_reviewers),
        "num_unique_train_reviewers": len(train_reviewers),
        "num_unique_test_reviewers": len(test_reviewers),
        "num_unique_cold_start_test_reviewers": len(cold_start_test_reviewers),
    }

    wandb.log(metadata_dict)
    train_reviews.to_parquet("train/train_reviews.parquet", index=False)
    train_listings.to_parquet("train/train_listings.parquet", index=False)
    train_reviewers.to_parquet("train/train_reviewers.parquet", index=False)
    test_reviews.to_parquet("test/test_reviews.parquet", index=False)
    test_listings.to_parquet("test/test_listings.parquet", index=False)
    test_reviewers.to_parquet("test/test_reviewers.parquet", index=False)
    cold_start_test_reviews.to_parquet("test/cold_start_test_reviews.parquet", index=False)
    cold_start_test_listings.to_parquet("test/cold_start_test_listings.parquet", index=False)
    cold_start_test_reviewers.to_parquet("test/cold_start_test_reviewers.parquet", index=False)

    dataset_art = wandb.Artifact(f"{start_date}_{end_date}_data", type="dataset")
    for dir in ["train", "test"]:
        dataset_art.add_dir(dir)
    wandb.log_artifact(dataset_art)

    # Modelling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    train_loader = prepare_data_loader(
        data=train_data,
        batch_size=config["train_batch_size"],
        num_neighbours=config["train_num_neighbours"],
    )
    test_loader = prepare_data_loader(
        data=test_data,
        batch_size=config["test_batch_size"],
        num_neighbours=config["test_num_neighbours"],
    )
    model = Model(hidden_channels=config["hidden_channels"], data=involved_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train and Evaluate Loss
    run_eval_wrapper = functools.partial(
        run_eval,
        involved_data,
        involved_reviewers,
        involved_listings2dict,
        reverse_involved_listings2dict,
        test_reviews,
        test_reviewers,
        wandb.config["rec_K"],
    )
    test_wrapper = functools.partial(
        test,
        test_loader,
        device
    )

    best_model_path = None
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    model_prefix = "./models"
    # Train and Evaluate Loss
    for epoch in range(1, wandb.config["epochs"] + 1):
        model_is_best = False
        train_loss = train(model, optimizer, train_loader, device)
        test_loss = test_wrapper(model)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            wandb.run.summary["best_train_loss"] = best_train_loss

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            wandb.run.summary["best_test_loss"] = best_test_loss
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

        model_path = f"{model_prefix}/{epoch}_model_state_dict.pt"
        torch.save(model.state_dict(), model_path)
        model_art = wandb.Artifact(f"{MODEL_NAME}_iteration_{iteration}", type="model")
        model_art.add_file(model_path)
        wandb.log_artifact(
            model_art,
            aliases=[LATEST_TAG],
        )
        if model_is_best:
            best_model_path = model_path

    production_model = None
    try:
        production_model_dir = wandb.use_artifact(
            f"{ENTITY_NAME}/{PROJECT_NAME}/{MODEL_REGISTRY_NAME}:{PRODUCTION_TAG}"
        ).download()
        production_model_path = f"{production_model_dir}/{os.listdir(production_model_dir)[0]}"
        production_model = load_model(
            production_model_path, wandb.config["hidden_channels"], involved_data
        )
        logger.info("Production model found")
    except Exception as e:
        print(e)
        logger.info("No production model found")


    contender_model = load_model(best_model_path, wandb.config["hidden_channels"], involved_data)
    contender_model_metrics = run_eval_wrapper(contender_model)
    contender_model_metrics_dict = {
        f"contender_model_u2i_hit_rate@{config['rec_K']}": contender_model_metrics[0],
        "contender_model_u2i_coverage": contender_model_metrics[1],
        f"contender_model_i2i_hit_rate@{config['rec_K']}": contender_model_metrics[2],
        "contender_model_i2i_coverage": contender_model_metrics[3],
    }
    wandb.log(contender_model_metrics_dict)

    if production_model == None:
        is_contender_model_better = True
    else:
        test_loss_by_production_model = test_wrapper(production_model)
        production_model_metrics = run_eval_wrapper(production_model)
        production_model_metrics_dict =  {
                "production_model_test_loss": test_loss_by_production_model,
                f"production_model_u2i_hit_rate@{config['rec_K']}": production_model_metrics[0],
                "production_model_u2i_coverage": production_model_metrics[1],
                f"production_model_i2i_hit_rate@{config['rec_K']}": production_model_metrics[2],
                "production_model_i2i_coverage": production_model_metrics[3],
        }
        wandb.log(production_model_metrics_dict)
        is_contender_model_better = (
            best_test_loss < production_model_metrics_dict['production_model_test_loss'] 
            and contender_model_metrics_dict[f"contender_model_u2i_hit_rate@{config['rec_K']}"] > production_model_metrics_dict[f"production_model_u2i_hit_rate@{config['rec_K']}"]
            and contender_model_metrics_dict[f"contender_model_i2i_hit_rate@{config['rec_K']}"] > production_model_metrics_dict[f"production_model_i2i_hit_rate@{config['rec_K']}"]
        )
    model_registry_art = wandb.Artifact(MODEL_REGISTRY_NAME, type="model_registry")
    model_registry_art.add_file(best_model_path)
    # By right, it should go through shadow deployment or A/B Testing first before pushing it to production
    wandb.log_artifact(
        model_registry_art,
        aliases=[PRODUCTION_TAG] if is_contender_model_better else [ARCHIVED_TAG],
    )
    # Save code
    wandb.run.log_code(
        "./", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb")
    )
    wandb.finish()
    logger.info("End of Retraining")
