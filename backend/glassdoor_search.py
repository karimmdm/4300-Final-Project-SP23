import pandas as pd
reviews = pd.read_csv("data1/glassdoor_top_firm_and_review.csv")

def match_job_title(job_title):
    # lookup in the df_over_50 dataframe for the closest job title, using case insensitive matching
    matching = reviews[reviews['job_title'].str.contains(job_title, case=False)].sort_values('average_firm_rating', ascending=False)
    # if there is no match, return None
    if matching.empty:
        return {
            "firm": "No Match",
            "date_review": "No Match",
            "job_title": "No Match",
            "review_title": "No Match",
            "review_body": "No Match",
            "average_firm_rating": "No Match",
            "overall_rating": "No Match",
            "headline": "No Match",
            "pros": "No Match",
            "cons": "No Match",
        }
    return matching.iloc[0]