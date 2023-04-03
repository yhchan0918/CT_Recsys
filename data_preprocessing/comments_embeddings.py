from multiprocessing import Pool
import json
import pandas as pd
from sentence_transformers import SentenceTransformer


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embed_model.max_seq_length = 512


def get_comment_embedding(data_dict):
    try:
        lang = data_dict["language"]
        id = data_dict["id"]
        host_name = data_dict["reviewer_host_name"]
        first_name = data_dict["reviewer_first_name"]
        print(id)
        if lang == "en":
            cmt = data_dict["comments"]
        else:
            cmt = data_dict["localized_comments"]

        if not cmt:
            if host_name or first_name:
                cmt = host_name + " " + first_name
            else:
                cmt = "Unknown User"

        embedding = (embed_model.encode(cmt)).tolist()

        return id, embedding
    except Exception as e:
        print(e)
        return id, []


if __name__ == "__main__":
    cols = [
        "id",
        "comments",
        "language",
        "localized_comments",
        "reviewer_host_name",
        "reviewer_first_name",
    ]
    reviews = pd.read_parquet("../../data/processed/reviews_with_interactions.parquet")[cols]
    tmp = reviews.copy()
    # 7.21
    iteration = 1
    unit = 200000
    start_idx = (iteration * unit) - unit
    end_idx = iteration * unit
    data_list = tmp.to_dict("records")[start_idx:end_idx]

    print(len(data_list))
    result_dict = {}
    result_list = []
    DIMENSION = 384
    for data in data_list:
        id, embeddings = get_comment_embedding(data)
        result_dict[id] = embeddings
        item_dict = {}
        item_dict["id"] = id

        for i in range(DIMENSION):
            try:
                item_dict[f"{i}"] = embeddings[i]
            except Exception as e:
                item_dict[f"{i}"] = None

        result_list.append(item_dict)

    write_dict_into_json(result_dict, f"./f/{iteration}.json")
    result_df = pd.DataFrame(result_list)
    result_df.to_parquet(f"./f/{iteration}.parquet", index=False)
