import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException

from selenium.webdriver.common.keys import Keys


class WebScraper:

    def __init__(self, filepath: str="product_links.csv"):
        data_df = self.load_data(filepath)
        self.original_urls = self.get_product_links_dict(data_df)
        self.urls = self.get_all_review_links_dict(self.original_urls)

        self.driver = webdriver.Chrome(ChromeDriverManager().install())

        self.data = {
            'productID': [],
            'productName': [],
            'productStarRating': [],
            'globalStarRatingCount': [],
            'ratingByFeatureAttributes': [],
            'reviewStars': [],
            'reviewHeader': [],
            'reviewText': [],
            'reviewHelpfulCount': [],
            'verifiedPurchase': []
        }

        self.scrape()
    
    
    def load_data(self, filepath: str):
        """
        Returns a pandas dataframe formed after reading the data in the entered filepath
        """
        return pd.read_csv(filepath)
    
    def get_product_links_dict(self, data_df) -> dict:
        """
        Takes a pandas dataframe of product links (data_df).
        Each product is assigned a unique product id (integers starting with 1 and increasing from there).
        A dictionary is formed in which the key is the product id, and the value is the url (product link)
        for the corresponding product. This dictionary is returned.
        """
        product_links_dict = dict()
        for index, df_row in data_df.iterrows():
            product_links_dict[index + 1] = df_row['productLinks']
        return product_links_dict
    
    def get_all_product_review_url_from_original_url(self, original_url: str) -> str:
        """
        Takes the link to a product on amazon.
        Modifies the link to get the link to the page on Amazon through which all of the customer
        reviews for the product can be accessed.
        Returns this new link.
        """
        tokenized = original_url.split("/")
        new_url = "/".join(token for token in tokenized[:4]) + "/product-reviews/"
        new_url += tokenized[5] + "/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
        return new_url
    
    def get_all_review_links_dict(self, product_links_dict) -> dict:
        """
        Takes a dictionary which maps product ids against product links (urls).
        Modifies each product link to get the link to the page on Amazon through which all of the customer
        reviews for the product can be accessed.
        Returns a new dictionary mapping from product ids to the corresponding "all customer review" product
        links.
        """
        all_review_links_dict = dict()
        for id in product_links_dict:
            all_review_links_dict[id] = self.get_all_product_review_url_from_original_url(product_links_dict[id])
        return all_review_links_dict

    
    def get_product_id(self, url_key: int) -> None:
        """
        Takes the product id for a certain product (url_key).
        Appends this value into the list mapped to by the key 'productId' in the self.data dictionary
        """
        self.data['productID'].append(url_key)
    
    def get_product_name(self, soup) -> None:
        """
        Takes a BeautifulSoup html parser for the html code for a given product's amazon page.
        Extracts the name of the product.
        Appends this value into the list mapped to by the key 'productName' in the self.data dictionary
        """
        product_name = soup.find('a', {'class': 'a-link-normal', 'data-hook': 'product-link'}).text.strip()
        self.data['productName'].append(product_name)
    
    def get_product_star_rating(self, soup) -> None:
        """
        Takes a BeautifulSoup html parser for the html code for a given product's amazon page.
        Extracts the star rating for the product.
        Appends this value into the list mapped to by the key 'productStarRating' in the self.data dictionary
        """
        try:
            star_rating = soup.find('span', {'class': 'a-size-medium a-color-base'}).text.strip()[:3]
            if '.' not in star_rating:
                star_rating = star_rating[0]
            self.data['productStarRating'].append(float(star_rating))
        except AttributeError:
            self.data['productStarRating'].append(None)
    
    def get_global_star_rating_count(self, soup) -> None:
        """
        Takes a BeautifulSoup html parser for the html code for a given product's amazon page.
        Extracts the number of people who have provided a star rating, using which the product's
        Global Star Rating has been determined.
        Appends this value into the list mapped to by the key 'globalStarRatingCount' in the self.data dictionary
        """
        try:
            gsrc_div = soup.find('div', {'class': 'a-row a-spacing-medium averageStarRatingNumerical'})
            global_star_rating_counter = gsrc_div.find('span').text.strip()
            end_index = global_star_rating_counter.index('g') - 1
            rating_count = int(global_star_rating_counter[:end_index].replace(',', ''))
            self.data['globalStarRatingCount'].append(rating_count)
        except AttributeError:
            self.data['globalStarRatingCount'].append(None)
    
    def get_review_stars(self, rating_card) -> None:
        """
        Takes an object representing a specific customer review.
        Extracts (for this specific customer review) the number of stars the customer gave the product.
        Appends this value into the list mapped to by the key 'reviewStars' in the self.data dictionary
        """
        try:
            rating_star = float(rating_card.find('span', {'class': 'a-icon-alt'}).text.strip()[:3])
            self.data['reviewStars'].append(rating_star)
        except AttributeError:
            self.data['reviewStars'].append(None)
    
    def get_review_header(self, rating_card) -> None:
        """
        Takes an object representing a specific customer review.
        Extracts the text in the heading of this customer review.
        Appends this value into the list mapped to by the key 'reviewHeader' in the self.data dictionary
        """
        try:
            rating_card_header_a = rating_card.find('a', {'class': 'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold'})
            review_header = rating_card_header_a.find('span').text.strip()
            self.data['reviewHeader'].append(review_header)
        except AttributeError:
            self.data['reviewHeader'].append(None)
    
    def get_review_text(self, rating_card) -> None:
        """
        Takes an object representing a specific customer review.
        Extracts the text in the body of this customer review.
        Appends this value into the list mapped to by the key 'reviewText' in the self.data dictionary
        """
        try:
            review_text_span_outer = rating_card.find('span', {'class': 'a-size-base review-text review-text-content'})
            review_text = review_text_span_outer.find('span').text.strip()
            self.data['reviewText'].append(review_text)
        except AttributeError as ae:
            self.data['reviewText'].append(None)
    
    def get_review_helpful_count(self, rating_card) -> None:
        """
        Takes an object representing a specific customer review.
        Extracts the number of people who flagged the customer review as being helpful.
        Appends this value into the list mapped to by the key 'reviewHelpfulCount' in the self.data dictionary
        """
        try:
            review_helpful_count = rating_card.find('span', {'class': 'a-size-base a-color-tertiary cr-vote-text'}).text.strip()
            end_index = review_helpful_count.index('p') - 1
            if review_helpful_count[:end_index] == 'One':
                self.data['reviewHelpfulCount'].append(1)
            else:
                helpful_count = int(review_helpful_count[:end_index].replace(',', ''))
                self.data['reviewHelpfulCount'].append(helpful_count)
        except AttributeError:
            self.data['reviewHelpfulCount'].append(0)
    
    def get_verified_purchase(self, rating_card) -> None:
        """
        Takes an object representing a specific customer review.
        Extracts a metric which defined whether or not Amazon has recorded the review as being written by
        someone who made a verified purchase of the product (extracts a boolean: True if it is from
        a verified purchase and False otherwise).
        Appends this value into the list mapped to by the key 'verifiedPurchase' in the self.data dictionary
        """
        try:
            verified_rating = rating_card.find('span', {'class': 'a-size-mini a-color-state a-text-bold'}).text.strip()
            if verified_rating == 'Verified Purchase':
                verified_purchase = True
            else:
                verified_purchase = False
        except AttributeError:
            verified_purchase = False
        self.data['verifiedPurchase'].append(verified_purchase)

    def capture_rating_by_feature_attributes(self, url_id) -> dict:
        """
        Takes the url (link) to a product on Amazon.
        Extracts the features mentioned and the corresponding ratings provided by Amazon in the 
        ("By Feature") section of the page.
        Returns a dictionary which maps the feature name (key) against the corresponding rating (value)
        """
        local_driver = webdriver.Chrome(ChromeDriverManager().install())

        local_driver.get(self.original_urls[url_id])
        page_content = local_driver.page_source
        soup = BeautifulSoup(page_content, 'html.parser')

        html = local_driver.find_element_by_tag_name('html')
        # Scroll down to load dynamic content so it can be scraped (Amazon only loads the html after you scroll
        # down to that part of the webpage)
        for _ in range(30):
            html.send_keys(Keys.PAGE_DOWN)
            for __ in range(25000000):
                pass
        try:
            try:
                ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
                l = WebDriverWait(local_driver, 15, ignored_exceptions = ignored_exceptions)\
                                    .until(EC.element_to_be_clickable((By.LINK_TEXT, 'See more')))
                l.click()
                soup = BeautifulSoup(local_driver.page_source, 'html.parser')
            except Exception:
                print("Could not click button initially. Trying again...")
                ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
                l = WebDriverWait(local_driver, 15, ignored_exceptions = ignored_exceptions)\
                                    .until(EC.element_to_be_clickable((By.LINK_TEXT, 'See more')))
                l.click()
                soup = BeautifulSoup(local_driver.page_source, 'html.parser')
                
            
            feature_divs = soup.findAll('div', {'class': 'a-fixed-right-grid-inner a-grid-vertical-align a-grid-center'})
            features = dict()
            for feature_div in feature_divs:
                feature_text = feature_div.find('span', {'class': 'a-size-base a-color-base'}).text.strip()
                feature_rating = float(feature_div.find('span', {'class': 'a-size-base a-color-tertiary'}).text.strip())
                features[feature_text] = feature_rating
            return features
        except Exception:
            print("Something went wrong in capture process!")
            return dict()
    
    def rating_by_feature_attributes(self, features) -> None:
        """
        Takes a dictionary (features) which maps feature names to corresponding rating scores.
        Appends this value into the list mapped to by the key 'ratingByFeatureAttributes' in the self.data dictionary
        """
        self.data['ratingByFeatureAttributes'].append(features)


    def get_data(self, soup, url_id) -> None:
        """
        Takes a BeautifulSoup html parser (soup) for the html code for a given product's amazon page.
        Also takes the url (link) to the Amazon page for a specific product.
        Scrapes all the required data from the given product url_id and appends it to the required
        locations in the self.data dictionary.
        """
        features = self.capture_rating_by_feature_attributes(url_id)
        while True:
            page_content = self.driver.page_source
            soup = BeautifulSoup(page_content, 'html.parser')

            rating_main_divs = soup.findAll('div', {'class': 'a-section review aok-relative'})
            for rating_card in rating_main_divs:
                self.get_review_stars(rating_card)
                self.get_review_header(rating_card)
                self.get_review_text(rating_card)
                self.get_review_helpful_count(rating_card)
                self.get_verified_purchase(rating_card)

                self.get_product_id(url_id)
                self.get_product_name(soup)
                self.get_product_star_rating(soup)
                self.get_global_star_rating_count(soup)
                self.rating_by_feature_attributes(features)
            
            try:
                self.driver.implicitly_wait(4)
                self.driver.refresh()
                ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
                l = WebDriverWait(self.driver, 15, ignored_exceptions=ignored_exceptions)\
                                    .until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'Next page')))
                l.click()
            except Exception as e:
                break
            self.driver.implicitly_wait(10)
    
    def write_data(self) -> None:
        """
        Exports the self.data dictionary into a CSV file.
        """
        df = pd.DataFrame(self.data)
        df = df.drop_duplicates(subset=['productID', 'productName', 'productStarRating', 'globalStarRatingCount',\
            'reviewStars', 'reviewHeader', 'reviewText', 'reviewHelpfulCount', 'verifiedPurchase'])
        # df = df.dropna()
        df.to_csv('../templates/static/data-files/review_data.csv')

    def scrape(self) -> None:
        """
        Calls all the functions that perform the webscraping operations on every product.
        Then calls the write_data function to export the scraped data as a CSV file.
        """
        for url_id in self.urls:
            self.driver.get(self.urls[url_id])
            page_content = self.driver.page_source
            soup = BeautifulSoup(page_content, 'html.parser')
            self.get_data(soup, url_id)
        self.write_data()
        