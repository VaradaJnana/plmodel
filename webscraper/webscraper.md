# Webscraper documentation

The code that performs the webscraping operations is contained within the class `WebScraper`, which is defined in `webscraper.py`.

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)
- [`selenium`](https://selenium-python.readthedocs.io/)
- [`bs4 (beautifulsoup)`](https://beautiful-soup-4.readthedocs.io/en/latest/)
- [`webdriver_manager`](https://pypi.org/project/webdriver-manager/)

Steps:
1. The class takes in the filepath to the file with the product links as an input and then loads this into a `pandas DataFrame`. This is done using the `load_data` function.
2. The `get_product_links_dict` function is called in order to create a dictionary with product IDs as keys and product links as values. This dictionary is stored in `self.original_urls`. These are the urls that we will actually use for the scraping process.
3. The `get_all_review_links_dict` function is called in order to modify the links in `self.original_urls` to get the links to the product pages with all the customer review links. We store this information in a dictionary called `self.urls`, which has product ids as keys and the "show all customer reviews" links as values.
4. We define a dictionary called `self.data`, which holds the format of the data that we will scrape from Amazon. Each of the features that we will collect is a key in the dictionary, and the values are lists that will contain the data.
5. The `scrape` function is called, which scrapes all the code. The steps involved in this are explained below


<br><br>
**Scraping steps:**

As part of the scraping process, for each of the products, we perform the following steps:
1. We use the Selenium webdriver to open the url for the product in Chrome.
2. We extract the HTML code from the website (via the Selenium driver).
3. We create a BeutifulSoup html parser on this extracted HTML code.
4. We call the `get_data` function to extract all the information we need and add it to `self.data`. Within `get_data`, on each page of product reviews, for every single review, we extract all the information we need by calling the following functions:
    * `get_review_stars`
    * `get_review_header`
    * `get_review_text`
    * `get_review_helpful_count`
    * `get_verified_purchase`
    * `get_product_id`
    * `get_product_name`
    * `get_product_star_rating`
    * `get_global_star_rating_count`
    * `rating_by_feature_attributes`

    While it exists, we keep finding and pressing the 'Next Page' button to keep fetching more customer reviews, and we scrape all the data from them using the aforementioned helped functions.
5. We call the `write_data` function, which exports the data in `self.data` into a CSV file called `review_data.csv` by first converting it to a `pandas DataFrame`. This CSV file is stored in the following location: `/templates/static/data-files/`. Every key in `self.data` is converted to a column heading. Then, every list item at the same index in the value lists are used to form the rows of this CSV file (each row of the CSV file represents a single customer review).


__Note__: There are docstrings (within `webscraper.py`) which explain what each function in the file `webscraper.py` does.