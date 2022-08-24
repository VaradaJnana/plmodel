# Product Ranking Documentation

There are two code files in this directory:
1. `product_attribute_ranking.py`
2. `product_ranker.py`

Both of these files are called at different points in the entire code flow: `product_attribute_ranking.py` is called from `run_before.py` to create a file called `attribute_product_mappings.csv`. This CSV file is stored in the location `templates/static/data-files`. The data in this file is an intermediate result which is used by `product_ranker.py` when it is called later.

On the other hand, `product_ranker.py` is called (from `data_files_loader.py`) when the dashboard is actually in use. If one makes use of the Product Ranker feature of the dashboard, then the code in `product_ranker.py` is run to create a CSV file which contains information on how customers rank different products on the market with respect to the product feature that was passed as an input to the form in the Product Ranker section of the webapp.

<hr>

**`product_attribute_ranking.py`:**

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)
- [`ast`](https://docs.python.org/3/library/ast.html) (Note: `ast` does not need to be installed; it comes by default with `python`)


The code in this file generates a CSV file by building upon the results that have already been stored in `product_attribute_descriptions.py`.

For each unique attribute mentioned in each row of the `topRelevantAttributes` column, the code in `product_attribute_ranking.py` identifies for which products this attribute was included in the `topRelevantAttributes` column. The extent to which customers like each product with respect to this attribute can be identified via the sentiment score for that attribute, which is stored in the dictionary in the `attributeScores` column. The set of products for which any given attribute was mentioned in the `topRelevantAttributes` column is then sorted in descending order of sentiment scores, and these are essentially the product ranks.

These results are then written to a CSV file, which has two columns: `attribute` and `productMapping`. Each row has a single attribute in the `attribute` column. The corresponding value in the `productMapping` column contains a list of tuples. Each tuple represents information on a single product for which this attribute was relevant. Each tuple contains the following 4 values:
1. product id
2. product name
3. descriptive words (used to describe the product with respect to the attribute)
4. sentiment score (for the product with respect to the attribute)
    - On a scale of -1.00 to +1.00 -- more positive scores indiciate that customers like the product more with respect to the product feature, and more negative scores indicate that customers dislike the product more with respect to the product feature. (Positive score: customers like the product with respect to the product feature; Negative score: customers like the product with respect to the product feature)

As mentioned above, these tuples are sorted in descending order of sentiment scores.

<hr>

**`product_ranker.py`:**

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)
- [`ast`](https://docs.python.org/3/library/ast.html) (Note: `ast` does not need to be installed; it comes by default with `python`)

The `product_ranker.py` file contains the `ProductRanker` class, which accepts a product feature (attribute) based on which customers want to rank products on the market. This file reads the data from `attribute_product_mappings.csv`. From this file, it extracts the product mapping for the chosen attribute. It then formats the data of this product mapping into a `pandas DataFrame`, with a separate column for each of the 4 values in the tuples stored in the product mapping. This `DataFrame` is stored in `self.result`, which can be accessed from `data_files_loader.py`, which is the file from which `product_ranker.py` is run.