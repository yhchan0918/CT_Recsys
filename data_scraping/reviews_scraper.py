import json
import requests
import time

API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
HEADERS = {"X-Airbnb-API-Key": API_KEY}


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


class ReviewsScraper:
    def __init__(self, listing_ids):
        self.listing_ids = listing_ids
        self.max_limit = 7
        self.API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
        self.HEADERS = {"X-Airbnb-API-Key": self.API_KEY}
        self.output_file_prefix = "../retry2"

    def build_request_url(self, listing_id, offset, limit):
        return f"https://www.airbnb.com/api/v3/PdpReviews?operationName=PdpReviews&locale=en&currency=USD&variables=%7B%22request%22%3A%7B%22fieldSelector%22%3A%22for_p3_translation_only%22%2C%22limit%22%3A{limit}%2C%22listingId%22%3A%22{listing_id}%22%2C%22offset%22%3A%22{offset}%22%2C%22showingTranslationButton%22%3Afalse%2C%22checkinDate%22%3A%222022-11-06%22%2C%22checkoutDate%22%3A%222022-11-12%22%2C%22numberOfAdults%22%3A%221%22%2C%22numberOfChildren%22%3A%220%22%2C%22numberOfInfants%22%3A%220%22%7D%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%226a71d7bc44d1f4f16cced238325ced8a93e08ea901270a3f242fd29ff02e8a3a%22%7D%7D"

    def get_reviews_count(
        self,
        listing_id,
    ):
        try:
            url = self.build_request_url(listing_id, 0, self.max_limit)
            r = requests.get(url=url, headers=self.HEADERS)
            data = r.json()
            reviews_count = data["data"]["merlin"]["pdpReviews"]["metadata"]["reviewsCount"]
            return reviews_count
        except Exception as e:
            print(listing_id, e)
            return 0

    def export_reviews(self, listing_id, reviews, reviews_count):
        temp = {}
        temp[listing_id] = {"reviews": reviews, "reviews_count": reviews_count}

        write_dict_into_json(temp, f"{self.output_file_prefix}/reviews_{listing_id}.json")

    def get_reviews(self):
        for listing_id in self.listing_ids:
            try:
                print(f"{listing_id} started")
                offset = 0
                max_reviews_count = self.get_reviews_count(listing_id)
                all_reviews = []
                self.export_reviews(listing_id, all_reviews, max_reviews_count)
                print(f"{listing_id} has {max_reviews_count} review count")
                if max_reviews_count == 0:
                    continue
                while offset <= max_reviews_count:
                    url = self.build_request_url(listing_id, offset, self.max_limit)
                    r = requests.get(url=url, headers=self.HEADERS)
                    data = r.json()
                    reviews = data["data"]["merlin"]["pdpReviews"]["reviews"]
                    all_reviews.extend(reviews)
                    print(len(all_reviews))
                    self.export_reviews(listing_id, all_reviews, max_reviews_count)
                    offset += self.max_limit
                    time.sleep(3)
            except Exception as e:
                print(listing_id, e)


f = open("../data/retry2_listing_ids.json")
retry_listing_ids = json.load(f)
print("Total No of listing id ", len(retry_listing_ids))
reviewsScraper = ReviewsScraper(retry_listing_ids)
reviewsScraper.get_reviews()
