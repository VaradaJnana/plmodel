# Report Results Documentation

This directory contains 3 files that contain code:
1. `report_results.py`
2. `attribute_negative_count_mapper.py`
3. `synonym_mapper.py`

`report_results.py` is the main file in the directory. Both `attribute_negative_count_mapper.py` and `synonym_mapper.py` are called from within `report_results.py` to help run the code within it.

<hr>


**`report_results.py`:**

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)
- [`ast`](https://docs.python.org/3/library/ast.html) (Note: `ast` does not need to be installed; it comes by default with `python`)
- [`textblob`](https://textblob.readthedocs.io/en/dev/)
- [`sklearn`](https://scikit-learn.org/stable/)
- [`spacy`](https://spacy.io/)
- [`nltk`](https://www.nltk.org/)

This file contains code that is used to generate report results in the form of CSV files. These CSV files contain data that is then later used by other parts of the code, or to directly display results on the dashboard.
The CSV files generated via the code in this file are:
1. `top_twenty_attributes.csv`: top 20 product features (attributes) that customers care about the most (calculated using the tf-idf algorithm)
2. `attribute_ranklist_complete.csv`: ranklist reflecting how much importance customers place in each of the product features (attributes) they spoke about (calculated using the tf-idf algorithm)
3. `amazon_suggested_attributes.csv`: product features (attributes) suggested by Amazon as being potentially relevant to customers
4. `product_attribute_descriptions_report.csv`: maps each product against the most relevant product features for that product. Also provides the words used to describe the product with respect to each of these product features, and a sentiment score for the product with respect to each of these product features (reflecting how much customers like the product with respect to that feature).
    - Sentiment scores: On a scale of -1.00 to +1.00 -- more positive scores indiciate that customers like the product more with respect to the product feature, and more negative scores indicate that customers dislike the product more with respect to the product feature. (Positive score: customers like the product with respect to the product feature; Negative score: customers like the product with respect to the product feature)

In order to help generate the data in these files in an effective manner, the code in `attribute_negative_count_mapper.py` and `synonym_mapper.py` is also called from within this file.

<hr>


**`attribute_negative_count_mapper.py`:**

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)

The code in this file is run to help generate some of the information required to create the file `attribute_negative_counts.csv`.
For each attribute (product feature) mentioned in customer reviews, this file records the number of products on the market for which the net sentiment of the product with respect to the attribute was negative.
(If the negative count for an attribute is 0, then that attribute is not included in the file).

<hr>


**`synonym_mapper.py`:**

_Dependencies:_
- [`string`](https://docs.python.org/3/library/string.html) (Note: `string` does not need to be installed; it comes by default with `python`)
- [`spacy`](https://spacy.io/)
- [`nltk`](https://www.nltk.org/)

The code in this file maintains a mapping from product features (attributes) that have already been mentioned in the text against the stemmed version of those attributes, and the synonyms of those attributes. Later, if the stemmed version or any of the synonyms of the word come up, they are replaced with the original attribute. This is done to reduce the total vocabulary count, and also to reduce the likelihood of two similar attributes being shown in the top 20 attributes, etc. (we try to merge the results of similar attributes together, since customers are essentially referring to the same product feature, just with different names).