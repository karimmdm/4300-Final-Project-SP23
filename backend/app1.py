import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# -------- Jensen's Imports, delete comment after merge pr
from collections import defaultdict
from collections import Counter
import json
import math
import string
import time
import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from IPython.core.display import HTML

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
# MYSQL_USER = "root"
# MYSQL_USER_PASSWORD = os.getenv("MYSQL_USER_PASSWORD")
# MYSQL_PORT = 3306
# MYSQL_DATABASE = "kardashiandb"

# mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# # Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db('/Users/jensen615/cs4300/careerfinder/init.sql')

app = Flask(__name__)
CORS(app)

def get_data():
    csv_files = [
        'data1/Data_Job_NY.csv',
        'data1/Data_Job_SF.csv',
    ]

    df = pd.DataFrame()

    for file in csv_files:
        df_temp = pd.read_csv(file)
        df = df.append(df_temp, ignore_index=True)

    jobs = df.drop(['Date_Posted', 'Valid_until', 'Job_Type'], axis=1)

    print(jobs.shape[0])
    return jobs

def inverted_index(jobs):
    """ Builds an inverted index from the job descriptions.
    
    Arguments
    =========
    
    jobs: pandas.dataframe
    
        Each row in this df has a 'Tokens' field 
        that contains the tokenized job descriptions
    
    Returns
    =======
    
    inverted_index: dict
        For each term, the index contains 
        a sorted list of tuples (job_id, count_of_term_in_doc)
        such that tuples with smaller job_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]
        
    """
    inv_index = defaultdict(list)
    i = 0
    
    for d in jobs['Tokens']:
        
        counts = Counter(d)        
        for k, v in counts.items():
            tup = (i, v)
            inv_index[k].append(tup)          
        i += 1
        
    return inv_index

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """ Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.
    
    Arguments
    =========
    
    inv_idx: an inverted index as above
    
    n_docs: int,
        The number of documents.
        
    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored. 
        Documents that appear min_df number of times should be included.
    
    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.
    
    Returns
    =======
    
    idf: dict
        For each term, the dict contains the idf value.
        
    """

    idf = {}
    
    for k, v in inv_idx.items():
        dft = len(v)
        if not (dft < min_df or (dft / n_docs) > max_df_ratio):   
            c = n_docs / (1 + dft)
            idf[k] = math.log2(c)
        
    return idf

def compute_doc_norms(index, idf, n_docs):
    """ Precompute the euclidean norm of each document.
    
    Arguments
    =========
    
    index: the inverted index as above
    
    idf: dict,
        Precomputed idf values for the terms.
    
    n_docs: int,
        The total number of documents.
    
    Returns
    =======
    
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    
    norms = np.zeros(n_docs)
    
    for k, v in index.items():
        idf_k = idf.get(k)

        if idf_k: 
            for doc_id, freq in v:
                norms[doc_id] += (idf_k * freq) **2

    return np.sqrt(norms)

def dot_scores(query_word_counts, index, idf):
    """ Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    
    Arguments
    =========
    
    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.
    
    index: the inverted index as above,
    
    idf: dict,
        Precomputed idf values for the terms.
    
    Returns
    =======
    
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    
    doc_scores = {}
    
    for k, v in query_word_counts.items():
        
        if v != 0: 
            i = index.get(k)
            idf_k = idf.get(k)
            
            if idf_k and i:
                for doc_id, freq in i:
                    dot = (v * idf_k) * (freq * idf_k)
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + dot 
        
    return doc_scores


tokenizer = RegexpTokenizer(r'\w+')
jobs = get_data()
jobs['Tokens'] = jobs['Job_Desc'].apply(lambda x: tokenizer.tokenize(x))

inv_idx = inverted_index(jobs)

# documents can be very long so we can use a large value here
# examine the actual DF values of common words like "the" to set these values
idf = compute_idf(inv_idx, jobs.shape[0], min_df=10, max_df_ratio=0.5)  

# prune the terms left out by idf
inv_idx = {key: val for key, val in inv_idx.items() if key in idf} 

doc_norms = compute_doc_norms(inv_idx, idf, jobs.shape[0])


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def index_search(query, index, idf, doc_norms, score_func=dot_scores, tokenizer=tokenizer):
    """ Search the collection of documents for the given query
    
    Arguments
    =========
    
    query: string,
        The query we are looking for.
    
    index: an inverted index as above
    
    idf: idf values precomputed as above
    
    doc_norms: document norms as computed above
    
    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
    
    tokenizer: a TreebankWordTokenizer
    
    Returns
    =======
    
    results, list of tuples (score, job_id)
        Sorted list of results such that the first element has
        the highest score, and `job_id` points to the document
        with the highest score.
    
    Note: 
        
    """
    
    q = tokenizer.tokenize(query)
    q_counts = Counter(q)
    doc_scores = score_func(q_counts, index, idf)

    q_norm = 0
    for k, v in q_counts.items():
        idf_k = idf.get(k)
        
        if idf_k: 
            q_norm += (idf_k * v) ** 2
            
    q_norm = math.sqrt(q_norm)
    
    results = []
    unique_jobs = set()
    
    for job_id, v in doc_scores.items():
        numer = v
        denom = q_norm * doc_norms[job_id]
        score = numer / denom
        
        tup = (jobs.iloc[job_id]['Job_title'], jobs.iloc[job_id]['Industry'])
        if not tup in unique_jobs:
            results.append((score, job_id))
            unique_jobs.add(tup)
        
    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results

def top10_results(query):

    print("#" * len(query))
    print(query)
    print("#" * len(query))

    count = 0
    results = index_search(query, inv_idx, idf, doc_norms)


    output = []
    for score, job_id in results:
        if count == 10:
            break
        count += 1
        
        result = {
            'Score': score,
            'Job Title': jobs.iloc[job_id]['Job_title'],
            'Industry': jobs.iloc[job_id]['Industry'],
            'Min Salary': jobs.iloc[job_id]['Min_Salary'],
            'Max Salary': jobs.iloc[job_id]['Min_Salary'],
        }

        result = json.dumps(result, default=np_encoder)
        output.append(result)
        
    return output



@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search")
def career_search():
    text = request.args.get("interest")
    return top10_results(text)


# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
# def sql_search(episode):
#     query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#     keys = ["id","title","descr"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys,i)) for i in data])

app.run(debug=True)