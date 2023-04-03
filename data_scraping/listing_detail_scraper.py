import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import pandas as pd
from multiprocessing import Pool
from enum import Enum


API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
API_ENDPOINT = (
    "https://www.airbnb.com/api/v3/StaysSearch?operationName=StaysSearch&locale=en&currency=USD"
)
HEADERS = {"X-Airbnb-API-Key": API_KEY}
LISTING_DETAIL_QUERY_PARAMS = "?check_in=2022-09-17&check_out=2022-09-18&enable_auto_translate=true&locale=en&country_override=US"


class LISTING_DETAIL_TYPE(str, Enum):
    NORMAL = "NORMAL"
    PLUS = "PLUS"
    LUXE = "LUXE"


RULES_DETAIL_PAGE = {
    "html": {
        "location": {"tag": "span", "class": "_9xiloll"},
        "title": {"tag": "h1", "class": "_fecoyn4"},
        "secondary_title": {"tag": "h2", "class": "_14i3z6h"},
        "price_per_night": {"tag": "span", "class": "_tyxjp1"},
    },
    "json": {
        "avg_rating": {
            "section_id": "REVIEWS_DEFAULT",
            "key": lambda x: x["section"]["overallRating"],
            "default_val": 0,
        },
        "num_of_reviews": {
            "section_id": "REVIEWS_DEFAULT",
            "key": lambda x: x["section"]["overallCount"],
            "default_val": 0,
        },
        "listing_type": {
            "section_id": "OVERVIEW_DEFAULT",
            f"{LISTING_DETAIL_TYPE.LUXE.value}_section_id": "TITLE_DEFAULT",
            "key": lambda x: x["section"]["subtitle"],
            f"{LISTING_DETAIL_TYPE.LUXE.value}_key": lambda x: x["section"]["shareSave"][
                "sharingConfig"
            ]["propertyType"],
        },
        "detail_items": {
            "section_id": "OVERVIEW_DEFAULT",
            f"{LISTING_DETAIL_TYPE.LUXE.value}_section_id": "OVERVIEW_LUXE",
            "key": lambda x: x["section"]["detailItems"],
            f"{LISTING_DETAIL_TYPE.LUXE.value}_key": lambda x: x["section"]["detailItems"],
        },
        "description": {
            "section_id": "DESCRIPTION_DEFAULT",
            f"{LISTING_DETAIL_TYPE.LUXE.value}_section_id": "UNSTRUCTURED_DESCRIPTION_LUXE",
            "key": lambda x: x["section"]["htmlDescription"]["htmlText"],
            f"{LISTING_DETAIL_TYPE.LUXE.value}_key": lambda x: json.loads(
                x["section"]["descriptionJsonString"]
            )["data"],
        },
        "category_rating": {
            "section_id": "REVIEWS_DEFAULT",
            "key": lambda x: x["section"]["ratings"],
        },
        "lat": {
            "section_id": "LOCATION_DEFAULT",
            "key": lambda x: x["section"]["lat"],
        },
        "lng": {
            "section_id": "LOCATION_DEFAULT",
            "key": lambda x: x["section"]["lng"],
        },
        "location_disclaimer": {
            "section_id": "LOCATION_DEFAULT",
            "key": lambda x: x["section"]["locationDisclaimer"],
        },
        "host_name": {
            "section_id": "HOST_PROFILE_DEFAULT",
            "key": lambda x: x["section"]["title"],
        },
        "joined_date_of_host": {
            "section_id": "HOST_PROFILE_DEFAULT",
            "key": lambda x: x["section"]["subtitle"],
        },
        "host_avatar_url": {
            "section_id": "HOST_PROFILE_DEFAULT",
            "key": lambda x: x["section"]["hostAvatar"]["avatarImage"]["baseUrl"],
        },
        "host_user_id": {
            "section_id": "HOST_PROFILE_DEFAULT",
            "key": lambda x: x["section"]["hostAvatar"]["userId"],
        },
        "host_features": {
            "section_id": "HOST_PROFILE_DEFAULT",
            "key": lambda x: x["section"]["hostFeatures"],
        },
        "host_tags": {
            "section_id": "HOST_PROFILE_DEFAULT",
            "key": lambda x: x["section"]["hostTags"],
        },
        "house_rules": {
            "section_id": "POLICIES_DEFAULT",
            "key": lambda x: x["section"]["houseRules"],
        },
        "additional_house_rules": {
            "section_id": "POLICIES_DEFAULT",
            "key": lambda x: x["section"]["additionalHouseRules"],
        },
        "listing_expectations": {
            "section_id": "POLICIES_DEFAULT",
            "key": lambda x: x["section"]["listingExpectations"],
        },
        "safety_expectations_and_amenities": {
            "section_id": "POLICIES_DEFAULT",
            "key": lambda x: x["section"]["safetyExpectationsAndAmenities"],
        },
        "sleeping_arrangement": {
            "section_id": [
                "SLEEPING_ARRANGEMENT_DEFAULT",
                "SLEEPING_ARRANGEMENT_WITH_IMAGES",
            ],
            "key": lambda x: x["section"]["arrangementDetails"],
        },
        "is_superhost": {
            "section_id": "TITLE_DEFAULT",
            "key": lambda x: extract_is_superhost(x["section"]["overviewItems"]),
            "default_val": False,
        },
        "amenities": {
            "section_id": ["AMENITIES_DEFAULT", "AMENITIES_PLUS", "AMENITIES_LUXE"],
            "key": lambda x: x["section"]["seeAllAmenitiesGroups"],
        },
        "hero_images": {
            "section_id": ["HERO_DEFAULT", "HERO_PLUS"],
            f"{LISTING_DETAIL_TYPE.LUXE.value}_section_id": "HERO_LUXE",
            "key": lambda x: x["section"]["previewImages"],
            f"{LISTING_DETAIL_TYPE.LUXE.value}_key": lambda x: x["section"]["heroMedia"],
        },
        "highlights": {
            "section_id": "HIGHLIGHTS_DEFAULT",
            "key": lambda x: x["section"]["highlights"],
        },
        "has_aircover": {
            "section_id": "AIRCOVER_PDP_BANNER",
            "key": lambda x: extract_has_aircover(x["section"]),
            "default_val": False,
        },
    },
}


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


def extract_element_data_from_html(soup, params):
    """Extracts data from a specified HTML element"""

    # 1. Find the right tag
    if "class" in params:
        elements_found = soup.find_all(params["tag"], params["class"])
    else:
        elements_found = soup.find_all(params["tag"])

    # 2. Extract text from these tags
    if "get" in params:
        element_texts = [el.get(params["get"]) for el in elements_found]
    else:
        element_texts = [el.get_text() for el in elements_found]

    # 3. Select a particular text or concatenate all of them
    tag_order = params.get("order", 0)
    if tag_order == -1:
        output = "**__**".join(element_texts)
    else:
        output = element_texts[tag_order]

    return output


def extract_listing_features_from_html(soup, rules, listing_id):
    """Extracts all features from the listing"""
    features_dict = {}
    for feature in rules:
        try:
            features_dict[feature] = extract_element_data_from_html(soup, rules[feature])
        except Exception as e:
            print(f"{listing_id}({feature}):", e)
            features_dict[feature] = None

    return features_dict


def find(pred, iterable):
    for element in iterable:
        if pred(element):
            return element
    return None


def extract_is_superhost(overview_items):
    for item in overview_items:
        if item["title"]:
            if item["title"].lower() == "superhost":
                return True


def extract_unstructured_description_luxe(description_list):
    return [data["text"] for data in description_list].join(" ")


def extract_has_aircover(section):
    if section:
        return True
    else:
        return False


def get_key_by_listing_detail_type(params, key, listing_detail_type):
    if listing_detail_type == LISTING_DETAIL_TYPE.NORMAL.value:
        return key
    else:
        special_key = f"{listing_detail_type}_{key}"
        if special_key in params:
            return special_key
        else:
            return key


def extract_element_data_from_json(sections, params, listing_detail_type):
    # 1. Find the right section
    if "section_id" in params:
        section_id_key = get_key_by_listing_detail_type(params, "section_id", listing_detail_type)
        if type(params[section_id_key]) is list:
            section = find(lambda x: x.get("sectionId") in params[section_id_key], sections)
        else:
            section = find(lambda x: x.get("sectionId") == params[section_id_key], sections)

    # 2. Find the right value by key
    output = None
    if "key" in params:
        key_key = get_key_by_listing_detail_type(params, "key", listing_detail_type)
        output = params[key_key](section)

    # 3. Replace return value with default value if there is one and output is empty
    if not output and "default_val" in params:
        output = params["default_val"]

    return output


def extract_listing_features_from_json(soup, rules, listing_detail_type, listing_id):
    """Extracts all features from the listing"""
    founds = soup.find_all("script", {"id": "data-deferred-state"})
    data = json.loads(founds[0].get_text())
    sections = data["niobeMinimalClientData"][0][1]["data"]["presentation"][
        "stayProductDetailPage"
    ]["sections"]["sections"]

    features_dict = {}
    for feature in rules:
        params = rules[feature]
        try:
            features_dict[feature] = extract_element_data_from_json(
                sections, params, listing_detail_type
            )
        except Exception as e:
            print(f"{listing_id}({feature}):", e)
            if "default_val" in params:
                fallback_val = params["default_val"]
            else:
                fallback_val = None
            features_dict[feature] = fallback_val

    return features_dict


def extract_listing_detail_type(url):
    if "/plus" in url:
        return LISTING_DETAIL_TYPE.PLUS.value
    elif "/luxury" in url:
        return LISTING_DETAIL_TYPE.LUXE.value
    else:
        return LISTING_DETAIL_TYPE.NORMAL.value


def extract_soup_js_and_url(listing_url, required_waiting_time=60):
    """Extracts HTML from JS pages: open, wait, click, wait, extract"""

    options = Options()
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    # if the URL is not valid - return an empty soup
    try:
        driver.get(listing_url)
    except Exception as e:
        print(f"Wrong URL: {listing_url}. {e}")
        return BeautifulSoup("", features="html.parser"), listing_url

    current_url = driver.current_url

    try:
        location_elem = WebDriverWait(driver, required_waiting_time).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, RULES_DETAIL_PAGE["html"]["location"]["class"])
            )
        )
        title_elem = WebDriverWait(driver, required_waiting_time).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, RULES_DETAIL_PAGE["html"]["title"]["class"])
            )
        )
        secondary_title_elem = WebDriverWait(driver, required_waiting_time).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, RULES_DETAIL_PAGE["html"]["secondary_title"]["class"])
            )
        )
        price_per_night_elem = WebDriverWait(driver, required_waiting_time).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, RULES_DETAIL_PAGE["html"]["price_per_night"]["class"])
            )
        )
        detail_api_response_elem = WebDriverWait(driver, required_waiting_time).until(
            EC.presence_of_element_located((By.XPATH, "//script[@id='datadeferred-state']"))
        )
    except Exception as e:
        # print(e)
        pass

    detail_page = driver.page_source
    driver.quit()

    return BeautifulSoup(detail_page, features="html.parser"), current_url


def build_url(listing_id):
    return f"https://www.airbnb.com/rooms/{listing_id}?{LISTING_DETAIL_QUERY_PARAMS}"


def scrape_detail_page(listing_id):
    """Scrapes the detail page and merges the result with basic features"""

    try:
        detail_url = build_url(listing_id)
        soup_detail, url = extract_soup_js_and_url(detail_url)
        listing_detail_type = extract_listing_detail_type(url)
        features_detailed_from_html = extract_listing_features_from_html(
            soup_detail, RULES_DETAIL_PAGE["html"], listing_id
        )
        features_detailed_from_json = extract_listing_features_from_json(
            soup_detail, RULES_DETAIL_PAGE["json"], listing_detail_type, listing_id
        )
        features = {}
        features["listing_detail_type"] = listing_detail_type
        features["url"] = url
        features["listing_id"] = listing_id

        features_all = {**features_detailed_from_html, **features_detailed_from_json, **features}
        # write_dict_into_json(features_all, f"../temp_detail/listing_detail_{listing_id}.json")
        return features_all
    except Exception as e:
        print(listing_id, e)
        return None


class Parser:
    def __init__(self, listing_ids):
        self.listing_ids = listing_ids

    def process_detail_pages(self):
        """Runs detail pages processing in parallel"""
        n_pools = 5
        with Pool(n_pools) as pool:
            result = pool.map(scrape_detail_page, self.listing_ids)
        pool.close()
        pool.join()

        self.all_features_list = result

    def parse(self):
        self.process_detail_pages()
        print(f"done")


if __name__ == "__main__":
    f = open(f"../../data/raw/listing/listing_ids/querys_output.json")
    file = json.load(f)
    listing_ids = file["all"]
    print(len(listing_ids))
    new_parser = Parser(listing_ids=listing_ids)
    t0 = time.time()
    new_parser.parse()
    print(time.time() - t0)
