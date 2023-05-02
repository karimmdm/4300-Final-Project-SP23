import pandas as pd
reviews = pd.read_csv("data1/glassdoor_top_firm_and_review.csv")

def match_job_title(job_title):
    # clean job title 
    job_title = job_title.strip()
    # remove any commas
    job_title = job_title.replace(',', '')
    # split on spaces
    job_words = job_title.split(' ')
    # for each word, remove any trailing s
    for i in range(len(job_words)):
        if job_words[i][-1] == 's':
            job_words[i] = job_words[i][:-1]
        # if job_words[i][-2:] == 'er':
        #     job_words[i] = job_words[i][:-2]
        # if job_words[i][-2:] == 'ie':
        #     job_words[i] = job_words[i][:-2]
        # if job_words[i][-10:] == 'ematician':
        #     job_words[i] = job_words[i][:-9]
        # if job_words[i][-10:] == 'ician':
        #     job_words[i] = job_words[i][:-5]
    if 'and' in job_words:
        job_words.remove('and')
    if 'the' in job_words:
        job_words.remove('the')

    # replace firm dashes with spaces
    reviews['firm'] = reviews['firm'].str.replace('-', ' ')
    # round score to 2 decimal places
    reviews['average_firm_rating'] = reviews['average_firm_rating'].round(2)
    
    # match any reviews that contain any of the words in the job title, case insensitive, with a preference for how many words match

    many_reviews = reviews[reviews['review_count_by_firm_role'] > 1]
    matching  = many_reviews[many_reviews['job_title'].str.contains('|'.join(job_words), case=False)].sort_values('average_firm_rating', ascending=False)
    
    # if there is no match, return None
    if matching.empty:
        return {
            "firm": "No Match",
            "date_review": "No Match",
            "job_title": "No Match",
            "average_firm_rating": "No Match",
            "overall_rating": "No Match",
            "headline": "No Match",
            "pros": "No Match",
            "cons": "No Match",
        }
    else:
        first = matching.iloc[0]
        res = {
            "firm": first['firm'],
            "date_review": first['date_review'],
            "job_title": first['job_title'],
            "average_firm_rating": first['average_firm_rating'],
            "overall_rating": first['overall_rating'],
            "headline": first['headline'],
            "pros": first['pros'],
            "cons": first['cons'],
        }
        return res