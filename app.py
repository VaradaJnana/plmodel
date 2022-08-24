from flask import Flask
from flask import render_template, redirect, url_for, request
from werkzeug.utils import secure_filename

import os
import sys
import webbrowser

print("Start loading!!")
from data_files_loader import DataFetcher
print("Done loading!!!")

from run_before import RunBefore

models_trained = None

app = Flask(__name__)
app._static_folder = os.path.abspath("templates/static")


p_a_d_product_info = "None"
product_ranker_info = "None"
features_set = None
chosen_feature = "N/A"
review_search_result = "None"
review_search_query = "N/A"
wordcloud_search_query = "N/A"
wordcloud_search_result = "No error"
scroll = "#"


@app.route("/")
def index():
    """
    This function contains the backend code which redirects the user to the
    appropriate page based on whether or not the models have already been
    trained in the past.
    If the models have already been trained, then the user is redirected
    to the dashboard page, since all the data has already been extracted,
    and the user can directly view and interact with the results through
    the dashboard page.
    If the models have not yet been trained (or if a new product type is
    being worked with), then the user is redirected to the load-data page,
    so they can submit a CSV file with product links, and then train the models.
    """
    if models_trained:
        # TODO: Have some way to go to some waiting page if the models are currently running
        global features_set
        features_set = DataFetcher().get_features_set()
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('load_data'))


@app.route("/load-data", methods=["GET", "POST"])
def load_data():
    """
    This function contains the backend code that governs the functioning of
    the load-data page.
    If the request to this route is a GET request, then the user is shown the
    load-data pagem, where they can enter the name of the product type, and
    upload a CSV file with product links.
    If a POST request is made to this route, then it accepts the product name
    type and the product links CSV file from the user. Once this data has been sent,
    it runs the code from run_before.py, and the scraping, absa model work,
    topic modelling, report generation, etc. takes place.
    Once this is complete, the user is automatically redirected to the dashboard
    page.
    """
    global models_trained

    if request.method == "POST":
        product_name = request.form.get("productName")
        file_to_be_scraped = request.files['fileinput']
        # NOTE: the line below is commented out for the demo; uncomment for actual use
        file_to_be_scraped.save(secure_filename("product_links.csv"))
        models_trained = True
        # NOTE: the line below is commented out for the demo; uncomment for actual use
        RunBefore(product_name=product_name.strip())
        global features_set
        features_set = DataFetcher().get_features_set()
        return redirect(url_for("index"))
    return render_template("layouts/load_data.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    """
    This function contains the backend code that governs the functioning
    of the dashboard route.
    Based on the POST requests that are made (depending on the data that
    the user requests from the dashboard on the frontend), this function
    calls the relevant functions from data_files_loader.py to fetch the
    data.
    This data is then served to the frontend HTML files, which in turn
    display it to the user, to display the result to their query.
    """
    if models_trained:
        global features_set
        features_set = DataFetcher().get_features_set()
    global p_a_d_product_info, product_ranker_info, chosen_feature, review_search_result, review_search_query, wordcloud_search_query, wordcloud_search_result, scroll
    data_fetcher = DataFetcher()
    product_rank_error = False

    top_twenty_attributes = data_fetcher.get_top_twenty_attributes()
    amazon_suggested_attributes = data_fetcher.get_amazon_suggested_attributes()

    products_list = data_fetcher.get_products_list()
    if request.method == "POST" and "p-a-d-productId" in request.form:
        # The user is requesting data from the Product Attribute Descriptions
        # functionality of the dashboard
        try:
            productId = int(request.form.get("p-a-d-productId"))
            p_a_d_product_info = data_fetcher.get_product_attribute_description_data(productId)
        except ValueError:
            pass
        scroll = "#product-attribute-descriptions"
    
    if request.method == "POST" and "productRankerFeature" in request.form:
        # The user is requesting data from the Product Ranker
        # functionality of the dashboard
        product_rank_feature = str(request.form.get("productRankerFeature")).replace(" ", "_")
        chosen_feature = product_rank_feature
        print("Chosen feature:", chosen_feature)
        try:
            product_ranker_info = data_fetcher.get_product_rank_info(product_rank_feature)
        except KeyError:
            product_rank_error = True
        scroll = "#product-ranker"
    
    if request.method == "POST" and "customerReviewSearchQuery" in request.form:
        # The user is requesting data from the Topic Modelling Customer Review Search
        # functionality of the dashboard
        review_search_query = str(request.form.get("customerReviewSearchQuery"))
        try:
            review_search_result = data_fetcher.get_customer_review_search_results(review_search_query)
        except ValueError:
            review_search_result = "Some word/words in your query were not present in even a single review"
        scroll = "#searching-customer-reviews"
    
    if request.method == "POST" and "wordcloudSearchQuery" in request.form:
        # The user is requesting data from the Topic Modelling Wordcloud Search
        # functionality of the dashboard
        wordcloud_search_query = str(request.form.get("wordcloudSearchQuery"))
        try:
            wordcloud_search_result = data_fetcher.get_wordcloud_search_results(wordcloud_search_query)
        except ValueError:
            wordcloud_search_result = "Some word/words in your query were not present in even a single review"
        scroll = "#searching-wordclouds"
    # print(wordcloud_search_result)
    
    improvement_areas_info = data_fetcher.get_market_improvement_areas_info()

    return render_template("layouts/dashboard.html", \
        top_twenty_attributes=top_twenty_attributes, \
        amazon_suggested_attributes=amazon_suggested_attributes, \
        products_list=products_list, \
        p_a_d_product_info=p_a_d_product_info, \
        features_set=features_set, \
        product_ranker_info=enumerate(product_ranker_info) if product_ranker_info != "None" else product_ranker_info, \
        product_rank_error=product_rank_error, \
        chosen_feature=chosen_feature, \
        improvement_areas_info=enumerate(improvement_areas_info), \
        review_search_result=review_search_result, \
        review_search_query=review_search_query, \
        wordcloud_search_query=wordcloud_search_query, \
        wordcloud_search_result=wordcloud_search_result, \
        improvements_filepath="file:///" + os.getcwd() + "/templates/static/data-files/improvement_areas.csv", \
        scroll=scroll
    )


@app.route("/improvements", methods=["POST"])
def improvements():
    """
    This function contains the backend code for the improvements route.
    This route only accepts POST requests. A POST request is sent to
    this route when the user clicks on the button which gives them the
    option to download a CSV file with all the improvement areas.
    When this POST request is made, the code in this function causes this
    file to be downloaded and automatically opened on the user's
    device.
    """
    webbrowser.open("file:///" + os.getcwd() + "/templates/static/data-files/improvement_areas.csv", new=2)
    return redirect(url_for('dashboard'))



if __name__ == '__main__':
    """
    main method from where this file can be called to start the functioning
    of the dashboard webapp from the terminal
    """
    models_trained = (sys.argv[1].lower() == 'models-trained')
    app.run(debug=True)