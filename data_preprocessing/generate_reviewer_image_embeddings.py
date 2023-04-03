from multiprocessing import Pool
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from urllib.parse import urlparse


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


def resnet(x: np.ndarray) -> np.ndarray:
    maps = model.predict(x)
    if np.prod(maps.shape) == maps.shape[-1] * len(x):
        return np.squeeze(maps)
    else:
        return maps.mean(axis=1).mean(axis=1)


def get_image_embeddings_from_url(data_dict):
    try:
        url = data_dict["reviewer_picture_url"]
        id = data_dict["reviewer_id"]
        url_parsed = urlparse(url)
        filepath = "./images/" + os.path.basename(url_parsed.path).split(".")[0] + ".png"
        driver = Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)
        driver.find_element(By.XPATH, "//img").screenshot(filepath)
        driver.quit()
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        embeddings = resnet(x)[0].tolist()
        result_dict = {"embed": embeddings}
        write_dict_into_json(result_dict, f"./embeddings/{id}.json")
        os.remove(filepath)
        return embeddings
    except:
        return []


# def get_image_embeddings_from_url(url):
#     try:
#         filename = wget.download(url)
#         img = image.load_img(filename, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         embeddings = resnet(x)
#         os.remove(filename)
#         return embeddings[0].tolist()
#     except:
#         return []


# def get_embeddings_by_row(row):
#     try:
#         url = row["reviewer_picture_url"]
#         embeddings = get_image_embeddings_from_url(url)
#         curr = pd.Series(embeddings[0])
#         resulting_row = pd.concat([row, curr])
#         return resulting_row
#     except:
#         return row


model = tf.keras.applications.ResNet50V2(include_top=False)


if __name__ == "__main__":
    reviewers = pd.read_parquet("../../data/processed/reviewers.parquet")
    tmp = reviewers.copy()
    data_list = tmp.to_dict("records")
    n_pools = 5
    with Pool(n_pools) as pool:
        result = pool.map(get_image_embeddings_from_url, data_list)
    pool.close()
    pool.join()

    embedding_dimension = 2048
    result_df = pd.DataFrame(result)
    result_df.columns = [f"f{i}" for i in range(embedding_dimension)]
    merge_df = pd.concat([tmp, result_df], axis=1)
    merge_df.to_parquet("reviewers_updated.parquet", index=False)
