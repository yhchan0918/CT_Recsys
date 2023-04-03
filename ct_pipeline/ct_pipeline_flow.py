from prefect import flow
from main_ct_run import run


@flow
def retrain_model_flow():
    run()


if __name__ == "__main__":
    retrain_model_flow()
