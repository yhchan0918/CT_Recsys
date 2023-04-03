import json
import requests
import time


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


class ReviewCountsScraper:
    def __init__(self, listing_ids):
        self.listing_ids = listing_ids
        self.max_limit = 7
        self.API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
        self.HEADERS = {"X-Airbnb-API-Key": self.API_KEY}
        self.output_file_prefix = "../temp_reviews"

    def build_request_url(self, listing_id, offset, limit):
        return f"https://www.airbnb.com/api/v3/PdpReviews?operationName=PdpReviews&locale=en&currency=USD&variables=%7B%22request%22%3A%7B%22fieldSelector%22%3A%22for_p3_translation_only%22%2C%22limit%22%3A{limit}%2C%22listingId%22%3A%22{listing_id}%22%2C%22offset%22%3A%22{offset}%22%2C%22showingTranslationButton%22%3Afalse%2C%22checkinDate%22%3A%222022-11-06%22%2C%22checkoutDate%22%3A%222022-11-12%22%2C%22numberOfAdults%22%3A%221%22%2C%22numberOfChildren%22%3A%220%22%2C%22numberOfInfants%22%3A%220%22%7D%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%226a71d7bc44d1f4f16cced238325ced8a93e08ea901270a3f242fd29ff02e8a3a%22%7D%7D"

    def get_reviews_counts(self):
        for listing_id in self.listing_ids:
            try:
                print(f"{listing_id} started")
                url = self.build_request_url(listing_id, 0, self.max_limit)
                r = requests.get(url=url, headers=self.HEADERS)
                data = r.json()
                reviews_count = data["data"]["merlin"]["pdpReviews"]["metadata"]["reviewsCount"]
            except Exception as e:
                reviews_count = "Error"
                print(listing_id, e)
            finally:
                output = {"listing_id": listing_id, "reviews_count": reviews_count}
                write_dict_into_json(
                    output, f"{self.output_file_prefix}/review_count_{listing_id}.json"
                )

    def parse(self):
        self.get_reviews_counts()


f = open("../data/scraped_data.json")
scraped_data = json.load(f)
listing_ids = scraped_data["all_unique_listing_ids"]
print("Total No of listing id ", len(listing_ids))
reviewCountsScraper = ReviewCountsScraper(listing_ids)
reviewCountsScraper.parse()
