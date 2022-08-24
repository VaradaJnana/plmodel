class TableStringCreator:

    def __init__(self, good: list, neutral: list, bad: list):
        """
        Takes 3 inputs:
        - good: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Good' with respect to the attribute in question
        - neutral: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Neutral' with respect to the attribute in question
        - bad: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Bad' with respect to the attribute in question
        

        Generates a formatted table with the content from good, neutral, and bad. This table is
        generated as a well formatted string. This string is stored in self.table_string
        """
        self.table_string = self.get_table_string(good, neutral, bad)
    

    def get_row_string(self, index: int, good: list, neutral: list, bad: list, len_good: int, \
                       len_neutral: int, len_bad: int) -> str:
        """
        Takes a range of inputs:
        - index: the index representing which row of the table is currently being generated
        - good: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Good' with respect to the attribute in question
        - neutral: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Neutral' with respect to the attribute in question
        - bad: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Bad' with respect to the attribute in question
        - len_good: the length of the list 'good'
        - len_neutral: the length of the list 'neutral'
        - len_bad: the length of the list 'bad'

        Creates and returns a string containing the entire content for the given row of the table
        (the string containing the space-formatted values that need to be put in the 'Good', 
        'Bad' and 'Neutral' columns)
        """
        good_string = good[index][1] if index < len_good else " "
        neutral_string = neutral[index][1] if index < len_neutral else " "
        bad_string = bad[index][1] if index < len_bad else " "
        row = f"\n {good_string : <30}  |  {bad_string : <30}  |  {neutral_string : <30}"
        return row

    def get_table_string(self, good: list, neutral: list, bad: list) -> str:
        """
        Takes 3 inputs:
        - good: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Good' with respect to the attribute in question
        - neutral: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Neutral' with respect to the attribute in question
        - bad: a list of strings, where each string holds the id and name of a product that
                has been classified as 'Bad' with respect to the attribute in question
        

        Generates a formatted table with the content from good, neutral, and bad. This table is
        generated in as a well formatted string. This string is then returned.
        """
        result = f"{'Good' : <30}  |  {'Bad' : <30}  |  {'Neutral' : <30}"
        len_good = len(good)
        len_bad = len(bad)
        len_neutral = len(neutral)
        max_len = max(len_good, len_bad, len_neutral)
        
        for i in range(max_len):
            row_string = self.get_row_string(i, good, neutral, bad, len_good, len_neutral, len_bad)
            result += row_string
        return result