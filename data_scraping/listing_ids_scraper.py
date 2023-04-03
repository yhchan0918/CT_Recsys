import json
import requests

API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
API_ENDPOINT = (
    "https://www.airbnb.com/api/v3/StaysSearch?operationName=StaysSearch&locale=en&currency=USD"
)
HEADERS = {"X-Airbnb-API-Key": API_KEY}


CATEGORY_TAGS = [
    677,
    675,
    8534,
    8225,
    4104,
    7769,
    8536,
    8528,
    5366,
    8192,
    8173,
    8187,
    8232,
    8159,
    8650,
    8099,
    8228,
    8230,
    8101,
    8157,
    634,
    1073,
    5731,
    7765,
    8043,
    8227,
    8175,
    789,
    8524,
    8535,
    8115,
    8148,
    8047,
    8525,
    8542,
    5635,
    8174,
    8538,
    8166,
    670,
    5348,
    8102,
    8522,
    8186,
    8526,
    8144,
    8256,
    8229,
    8255,
    8521,
    8543,
    8226,
    8176,
]
QUERYS = [
    "Thailand",
    "United Kingdom",
    "Malaysia",
    "Europe",
    "Australia",
    "Indonesia",
    "United States",
    "New York, NY",
    "Vietnam",
    "Singapore",
    "Japan",
    "Korea",
    "Paris" "Tokyo, Japan",
    "Canada" "Spain",
    "Romania",
    "Mexico",
    "Vanuatu",
    "Colombia",
    "Luxembourg",
    "Scotland",
    "Brazil",
    "France",
    "India",
    "Kenya",
    "Netherlands",
    "Italy",
    "Portugal",
]
MAIN_LISTINGS = []


def write_dict_into_json(dictionary, filename):
    with open(filename, "w") as write_file:
        json.dump(dictionary, write_file, indent=4)


class StaysSearchScraper:
    def __init__(self, query):
        self.query = query
        self.max_page = 20
        self.API_ENDPOINT = "https://www.airbnb.com/api/v3/StaysSearch?operationName=StaysSearch&locale=en&currency=USD"
        self.API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
        self.HEADERS = {"X-Airbnb-API-Key": self.API_KEY}
        self.listings = []

    def build_request_payload(self, items_offset, section_offset):
        return {
            "operationName": "StaysSearch",
            "variables": {
                "isInitialLoad": "true",
                "hasLoggedIn": "true",
                "cdnCacheSafe": "false",
                "source": "EXPLORE",
                "exploreRequest": {
                    "metadataOnly": "false",
                    "version": "1.8.3",
                    "tabId": "home_tab",
                    "refinementPaths": ["/homes"],
                    "priceFilterInputType": 0,
                    "datePickerType": "flexible_dates",
                    "source": "structured_search_input_header",
                    "searchType": "unknown",
                    "flexibleTripLengths": ["one_week"],
                    "federatedSearchSessionId": "7d4abc19-5d13-4b1e-9390-4aaa3075b565",
                    "itemsOffset": items_offset,
                    "sectionOffset": section_offset,
                    "query": self.query,
                    "itemsPerGrid": 20,
                    "cdnCacheSafe": "false",
                    "treatmentFlags": [
                        "flex_destinations_june_2021_launch_web_treatment",
                        "new_filter_bar_v2_fm_header",
                        "new_filter_bar_v2_and_fm_treatment",
                        "merch_header_breakpoint_expansion_web",
                        "flexible_dates_12_month_lead_time",
                        "storefronts_nov23_2021_homepage_web_treatment",
                        "lazy_load_flex_search_map_compact",
                        "lazy_load_flex_search_map_wide",
                        "im_flexible_may_2022_treatment",
                        "im_flexible_may_2022_treatment",
                        "flex_v2_review_counts_treatment",
                        "search_add_category_bar_ui_ranking_web_aa",
                        "flexible_dates_options_extend_one_three_seven_days",
                        "super_date_flexibility",
                        "micro_flex_improvements",
                        "micro_flex_show_by_default",
                        "search_input_placeholder_phrases",
                        "pets_fee_treatment",
                    ],
                    "screenSize": "large",
                    "isInitialLoad": "true",
                    "hasLoggedIn": "true",
                },
                "staysSearchM2Enabled": "false",
                "staysSearchM6Enabled": "false",
            },
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "a743098b24b86de94a0af99620b9f6a35664ca58e1dd0afdc81802dd157d1155",
                }
            },
        }

    def get_total_inventory_count(self):
        json_payload = self.build_request_payload(0, 1)
        r = requests.post(url=self.API_ENDPOINT, json=json_payload, headers=self.HEADERS)
        data = r.json()
        sections = data["data"]["presentation"]["explore"]["sections"]["sections"]
        pagination_section = None
        for section in sections:
            if section["sectionComponentType"] == "EXPLORE_NUMBERED_PAGINATION":
                pagination_section = section

        self.total_inventory_count = int(pagination_section["section"]["totalInventoryCount"])

    def get_list_of_listing_id(self):
        items_offset = 0
        section_offset = 1
        while items_offset <= self.total_inventory_count:
            try:
                json_payload = self.build_request_payload(items_offset, section_offset)
                r = requests.post(url=self.API_ENDPOINT, json=json_payload, headers=self.HEADERS)
                data = r.json()
                sections = data["data"]["presentation"]["explore"]["sections"]["sections"]
                listing_section = None
                for section in sections:
                    if section["sectionComponentType"] == "EXPLORE_SECTION_WRAPPER":
                        listing_section = section

                listings = listing_section["section"]["child"]["section"]["items"]
                for listing in listings:
                    self.listings.append(listing["listing"]["id"])
            except Exception as error:
                print(error)
            finally:
                items_offset += self.max_page
                if items_offset == 300 or items_offset == 600 or items_offset == 900:
                    section_offset += 1

        self.listings = list(set(self.listings))


class ExploreSectionScraper:
    def __init__(self, category_tag):
        self.category_tag = category_tag
        self.max_page = 20
        self.max_items_offset = 500
        self.API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
        self.HEADERS = {"X-Airbnb-API-Key": self.API_KEY}
        self.listings = []

    def build_request_url(self, category_tag, items_offset, section_offset):
        return f"https://www.airbnb.com/api/v3/ExploreSections?operationName=ExploreSections&locale=en&currency=USD&variables=%7B%22isInitialLoad%22%3Atrue%2C%22hasLoggedIn%22%3Atrue%2C%22cdnCacheSafe%22%3Afalse%2C%22source%22%3A%22EXPLORE%22%2C%22exploreRequest%22%3A%7B%22metadataOnly%22%3Afalse%2C%22version%22%3A%221.8.3%22%2C%22tabId%22%3A%22all_tab%22%2C%22refinementPaths%22%3A%5B%22%2Fhomes%22%5D%2C%22searchMode%22%3A%22flex_destinations_search%22%2C%22itemsPerGrid%22%3A20%2C%22cdnCacheSafe%22%3Afalse%2C%22treatmentFlags%22%3A%5B%22flex_destinations_june_2021_launch_web_treatment%22%2C%22merch_header_breakpoint_expansion_web%22%2C%22flexible_dates_12_month_lead_time%22%2C%22storefronts_nov23_2021_homepage_web_treatment%22%2C%22lazy_load_flex_search_map_compact%22%2C%22lazy_load_flex_search_map_wide%22%2C%22im_flexible_may_2022_treatment%22%2C%22im_flexible_may_2022_treatment%22%2C%22flex_v2_review_counts_treatment%22%2C%22search_add_category_bar_ui_ranking_web_aa%22%2C%22flexible_dates_options_extend_one_three_seven_days%22%2C%22super_date_flexibility%22%2C%22micro_flex_improvements%22%2C%22micro_flex_show_by_default%22%2C%22search_input_placeholder_phrases%22%2C%22pets_fee_treatment%22%5D%2C%22screenSize%22%3A%22large%22%2C%22isInitialLoad%22%3Atrue%2C%22hasLoggedIn%22%3Atrue%2C%22flexibleTripLengths%22%3A%5B%22one_week%22%5D%2C%22locationSearch%22%3A%22MIN_MAP_BOUNDS%22%2C%22categoryTag%22%3A%22Tag%3A{category_tag}%22%2C%22priceFilterInputType%22%3A0%2C%22priceFilterNumNights%22%3A5%2C%22itemsOffset%22%3A{items_offset}%2C%22sectionOffset%22%3A{section_offset}%2C%22federatedSearchSessionId%22%3A%2253c99239-03a7-4259-a5fe-22d1bb362ed3%22%7D%2C%22gpRequest%22%3A%7B%22expectedResponseType%22%3A%22INCREMENTAL%22%7D%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%2247aebb6c939057cf8d630a02a99021e11ab88eb0c9b889a21cb9a4c720cabf07%22%7D%7D"

    def get_list_of_listing_id(self):
        items_offset = 0
        section_offset = 0
        while items_offset <= self.max_items_offset:
            try:
                url = self.build_request_url(self.category_tag, items_offset, section_offset)
                r = requests.get(url=url, headers=self.HEADERS)
                data = r.json()
                items = data["data"]["presentation"]["explore"]["sections"]["responseTransforms"][
                    "transformData"
                ][0]["sectionContainer"]["section"]["child"]["section"]["items"]
                for item in items:
                    self.listings.append(item["listing"]["id"])
            except Exception as error:
                print(error)
            finally:
                items_offset += self.max_page

        self.listings = list(set(self.listings))


for query in QUERYS:
    print(f"{query} started")
    scraper = StaysSearchScraper(query)
    scraper.get_total_inventory_count()
    scraper.get_list_of_listing_id()
    listings = scraper.listings
    MAIN_LISTINGS.extend(listings)
    querys_output = {"all": MAIN_LISTINGS, "unique": list(set(MAIN_LISTINGS))}
    write_dict_into_json(querys_output, "querys_output.json")
    print(f"{query} ended")

for category_tag in CATEGORY_TAGS:
    print(f"{category_tag} started")
    scraper = ExploreSectionScraper(category_tag)
    scraper.get_list_of_listing_id()
    listings = scraper.listings
    MAIN_LISTINGS.extend(listings)
    category_tags_output = {"all": MAIN_LISTINGS, "unique": list(set(MAIN_LISTINGS))}
    write_dict_into_json(category_tags_output, "category_tags_output.json")
    print(f"{category_tag} ended")


print(f"Script finished")
