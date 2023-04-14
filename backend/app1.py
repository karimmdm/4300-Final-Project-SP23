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
import sys
from helpers import  OnetCsvHandler

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

def get_data_legacy():
    csv_files = [
        'data1/Data_Job_NY.csv',
        'data1/Data_Job_SF.csv',
    ]

    df = pd.DataFrame()

    for file in csv_files:
        df_temp = pd.read_csv(file)
        df = df.append(df_temp, ignore_index=True)

    jobs = df.drop(['Date_Posted', 'Valid_until', 'Job_Type'], axis=1)

    # print(jobs.shape[0])
    return jobs

def get_data():
    csv_handler = OnetCsvHandler.OnetCsvHandler()
    file = os.path.join(os.environ['ROOT_PATH'], "backend/helpers/raw_data")
    csv_handler.bulk_update(file)
    jobs = csv_handler.data()

    # print(type(jobs)
    return jobs

# --------------- EXAMPLE STRUCTURE OF ONET DATA --------------- #
'''
45-2041.00 {
    'occupation': 'Graders and Sorters, Agricultural Products', 
    'values': [('Working_Conditions', None), ('Support', None)], 
    'knowledge': [('Administrative', 16), ('Engineering_and_Technology', 23), ('Administration_and_Management', 32), ('English_Language', '51'), ('Biology', '10'), ('Philosophy_and_Theology', '8'), ('Psychology', 10), ('Customer_and_Personal_Service', 28), ('Law_and_Government', 9), ('Computers_and_Electronics', 25), ('Medicine_and_Dentistry', '17'), ('Building_and_Construction', '8'), ('Food_Production', '45'), ('Public_Safety_and_Security', 33), ('History_and_Archeology', '6'), ('Economics_and_Accounting', 7), ('Geography', 6), ('Therapy_and_Counseling', 9), ('Chemistry', '18'), ('Mathematics', 27), ('Communcations_and_Media', 12), ('Physics', '14'), ('Telecommunications', 6), ('Sociology_and_Anthropology', 5), ('Fine_Arts', '5'), ('Sales_and_Marketing', 21), ('Personnel_and_Human_Resources', 24), ('Design', 5), ('Mechanical', 50), ('Education_and_Training', 43), ('Production_and_Processing', 66), ('Foreign_Language', 34), ('Transportation', 24)], 
    'interests': [('Realistic', None)], 
    'cross-skills': [('Programming', 0), ('Persuasion', 22), ('Equipment Maintenance', 0), ('Quality Control Analysis', 22), ('Operation and Control', 13), ('Service Orientation', 22), ('Instructing', 16), ('Coordination', 38), ('Judgment and Decision Making', 31), ('Troubleshooting', 13), ('Installation', 0), ('Management of Financial Resources', 3), ('Equipment Selection', 0), ('Complex Problem Solving', 28), ('Negotiation', 19), ('Operations Monitoring', 16), ('Systems Evaluation', 16), ('Management of Personnel Resources', 10), ('Management of Material Resources', 3), ('Social Perceptiveness', 28), ('Operations Analysis', 0), ('Technology Design', 0), ('Repairing', 0), ('Systems Analysis', 22), ('Time Management', 31)] 
}

inverted_index[term] = [(job1, tf1), (job2, tf2), ...]
'''

def inverted_index(jobs):
    """ Builds an inverted index from the job descriptions.
    
    Arguments
    =========
    
    jobs: dict
        keys: job code
        vals: dictionary e.g.
            {
                'occupation': 'Graders and Sorters, Agricultural Products', 
                'values': [('Working_Conditions', None), ('Support', None)], 
                'knowledge': [('Administrative', 16), ('Engineering_and_Technology', 23), ('Administration_and_Management', 32), ('English_Language', '51'), ('Biology', '10'), ('Philosophy_and_Theology', '8'), ('Psychology', 10), ('Customer_and_Personal_Service', 28), ('Law_and_Government', 9), ('Computers_and_Electronics', 25), ('Medicine_and_Dentistry', '17'), ('Building_and_Construction', '8'), ('Food_Production', '45'), ('Public_Safety_and_Security', 33), ('History_and_Archeology', '6'), ('Economics_and_Accounting', 7), ('Geography', 6), ('Therapy_and_Counseling', 9), ('Chemistry', '18'), ('Mathematics', 27), ('Communcations_and_Media', 12), ('Physics', '14'), ('Telecommunications', 6), ('Sociology_and_Anthropology', 5), ('Fine_Arts', '5'), ('Sales_and_Marketing', 21), ('Personnel_and_Human_Resources', 24), ('Design', 5), ('Mechanical', 50), ('Education_and_Training', 43), ('Production_and_Processing', 66), ('Foreign_Language', 34), ('Transportation', 24)], 
                'interests': [('Realistic', None)], 
                'cross-skills': [('Programming', 0), ('Persuasion', 22), ('Equipment Maintenance', 0), ('Quality Control Analysis', 22), ('Operation and Control', 13), ('Service Orientation', 22), ('Instructing', 16), ('Coordination', 38), ('Judgment and Decision Making', 31), ('Troubleshooting', 13), ('Installation', 0), ('Management of Financial Resources', 3), ('Equipment Selection', 0), ('Complex Problem Solving', 28), ('Negotiation', 19), ('Operations Monitoring', 16), ('Systems Evaluation', 16), ('Management of Personnel Resources', 10), ('Management of Material Resources', 3), ('Social Perceptiveness', 28), ('Operations Analysis', 0), ('Technology Design', 0), ('Repairing', 0), ('Systems Analysis', 22), ('Time Management', 31)] 
            }
    
    Returns
    =======
    
    inverted_index: dict
        For each term (skill), the index contains 
        a sorted list of tuples (job_id, importance-score)
        such that tuples with smaller job_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]
        
    """
    inv_idx = defaultdict(list)

    # i = 0
    # for k, v in jobs.items():
    #     if i < 2:
    #         print(k, v)
    #     i += 1

    for doc, d in jobs.items():

        # used for debugging problem with d.get('cross-skills')
        # print(doc, d.keys())

        skills = d.get('cross-skills')
        if skills:
            for skill, tf in skills:
                skill = skill.lower().replace('_', ' ')
                
                try: 
                    score = int(tf)
                    if score > 20:
                        inv_idx[skill].append((doc, score))
                except:
                    pass

        knowledge = d.get('knowledge')
        if knowledge:
            for skill, tf in knowledge:
                skill = skill.lower().replace('_', ' ')

                try: 
                    score = int(tf)
                    if score > 20:
                        inv_idx[skill].append((doc, score))
                except:
                    pass
                

    return inv_idx

def jobToIdx(jobs):
    job_to_idx = {}
    
    counter = 0
    for id in jobs.keys():
        job_to_idx[id] = counter
        counter += 1

    return job_to_idx

def skillToIdx(inv_idx):
    skill_to_idx = {}
    
    counter = 0
    for s in inv_idx.keys():
        skill_to_idx[s] = counter
        counter += 1

    return skill_to_idx

# def compute_idf_legacy(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
#     """ Compute term IDF values from the inverted index.
#     Words that are too frequent or too infrequent get pruned.
    
#     Arguments
#     =========
    
#     inv_idx: an inverted index as above
    
#     n_docs: int,
#         The number of documents.
        
#     min_df: int,
#         Minimum number of documents a term must occur in.
#         Less frequent words get ignored. 
#         Documents that appear min_df number of times should be included.
    
#     max_df_ratio: float,
#         Maximum ratio of documents a term can occur in.
#         More frequent words get ignored.
    
#     Returns
#     =======
    
#     idf: dict
#         For each term, the dict contains the idf value.
        
#     """

#     idf = {}
    
#     # key: terms, values: list tuples (job_id, score freq)
#     for k, v in inv_idx.items():
#         dft = len(v)

#         print("length of tuples")
#         print(dft)
#         print("max ratio")
#         print(dft/n_docs)
#         print()

#         # makes assumption around scale of #
#         # if not (dft < min_df or (dft / n_docs) > max_df_ratio):   
#         c = n_docs / (1 + dft)
#         idf[k] = math.log2(c)
        
#     return idf

def compute_idf(index, n_docs, min_df=10, max_df_ratio=0.95):
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
    
    # key: skills, values: list of (job_id, score) tuples 
    for skill, postings in index.items():
        # dft: document frequency for this skill (term)
        dft = len(postings)

        # print(f"length of tuples for: {skill}")
        # print(dft)
        # print("dft/N ratio")
        # print(dft/n_docs)
        # print()

        # makes assumption around scale of #
        # if not (dft < min_df or (dft / n_docs) > max_df_ratio):   
        c = n_docs / (1 + dft)
        idf[skill] = math.log2(c)
        
    return idf

# [UPDATED: JA] this computes tf-idf (or skills-score_ijf)
#   - At this moment, we don't need this since we calculate tf-idf efficiently using a term-by-term approach
#   - within the compute_doc_norms function
#   - This is what JA, KC, RK were working on 04/12/23

# def compute_tf_idf(inv_idx, jobs, min_df=10, max_df_ratio=0.95):    
#     n_careers = len(jobs.items())
#     n_skills = len(inv_idx.items())

#     tf_idf = np.zeros((n_skills, n_careers))

#     for idx, skill in enumerate(inv_idx):
#         career_tups = inv_idx.get(skill)
#         len_ct = len(career_tups)
#         # print("index: ")
#         # print(idx)
#         # print()
#         # print("skill: ")
#         # print(skill)
#         # print()
#         # print("careers: ")
#         # print(career_tups)
#         # print()

#         for job, score in career_tups:
#             numer = 1 + n_careers
#             denom = 1 + len_ct
#             idf =  math.log2(numer / denom) + 1
#             cur = float(score) * idf             
#             c_id = job_to_idx.get(job)
#             tf_idf[idx][c_id] = cur

#     return tf_idf


def compute_doc_norms(index, idf, n_docs):
    """ Precompute the euclidean norm of each document.
    
    Arguments
    =========
    
    index: the inverted index as above
    
    idf: dict,
        Precomputed idf values for the terms.
    
    n_docs: int,
        The total number of jobs
    
    Returns
    =======
    
    norms: np.array, size: n_docs
        norms[i] = the norm of job i.
    """
    norms = np.zeros(n_docs)

    for skill, career_tups in index.items():

        idf_k = idf.get(skill)
        if idf_k:

            for job_id, score in career_tups:
                job_idx = job_to_idx.get(job_id)
                norms[job_idx] += (float(score) * idf_k) **2
    
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
    
    for term, freq in query_word_counts.items():
        
        # just a sanity check, can try w/ and w/out it
        if freq != 0:
            # get list of career tuples for given skill
            career_tups = index.get(term)
            # get idf score for given skill
            idf_k = idf.get(term)
            
            # make sure that this term wasn't pruned from inv_idx and that career_tups isn't empty
            if idf_k and career_tups:

                for job_id, score in career_tups:

                    dot = (freq * idf_k) * (score * idf_k)
                    doc_scores[job_id] = doc_scores.get(job_id, 0) + dot 
        
    return doc_scores


tokenizer = RegexpTokenizer(r'\w+')
jobs = get_data()
# jobs['Tokens'] = jobs['Job_Desc'].apply(lambda x: tokenizer.tokenize(x))

print("jobs length")
print(len(jobs.items()))
print()


# key: skill, val: list of (job_id, score) tuples
inv_idx = inverted_index(jobs)
# to display inv_idx
print("inv_idx after creation: ")
print("len inv_idx: ", len(inv_idx.items()))

# i = 0
# for k, v in inv_idx.items():
#     if i < 2:
#         print(k, v)
#         print()
#     i += 1
# i = 0
# print()


# job to index dictionary (needed for computing doc norms)
# key: job_id, val: index
job_to_idx = jobToIdx(jobs)

# key: skill, val: index
skill_to_idx = skillToIdx(inv_idx)


n_docs = len(jobs.items())
idf = compute_idf(inv_idx, n_docs)
# to display idf dict
print("idf after creation: ")
print("len idf: ", len(idf.items()))
print("should be same len as inv_idx")

# i = 0
# for k, v in idf.items():
#     if i < 2:
#         print(k, v)
#         print()
#     i += 1
# i = 0

# for k, v in idf.items():
#     print(k, v)
#     print()
# print()


# update inv_idx: prune the terms left out by idf (not needed ATM since all terms have an idf score given current implementation)
# inv_idx = {key: val for key, val in inv_idx.items() if key in idf}
# print("inv_idx items after: ")
# print(inv_idx.keys())
# print()

doc_norms = compute_doc_norms(inv_idx, idf, n_docs)
# print("doc_norms after creation: ")
# print("len doc_norms: ", len(doc_norms))
# print(doc_norms)
# print()


# [UPDATE: JA] This is where I left off 04/13/23; everything up to this point is clean and usable
# all that is left is to update index_search with the updated inv_idx, idf, and doc_norms

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

    # ---- This section is for normalizing query (q_norm) ---- #
    q_norm = 0
    # key = term, val = count
    for k, v in q_counts.items():
        idf_k = idf.get(k)
        
        if idf_k: 
            q_norm += (idf_k * v) ** 2

    q_norm = math.sqrt(q_norm)
    # ---- This section is for normalizing query (q_norm) ---- #
    
    results = []
    
    # print(len(doc_scores.items()))
    for job_id, v in doc_scores.items():
        job_idx = job_to_idx.get(job_id)

        numer = v
        denom = q_norm * doc_norms[job_idx]
        sim_score = numer / denom
        results.append((sim_score, job_id))
        
    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results

def top10_results(query):

    print("#" * len(query))
    print(query)
    print("#" * len(query))

    results = index_search(query, inv_idx, idf, doc_norms)
    # print(results)

    count = 0
    output = []
    for score, job_id in results:
        if count == 10:
            break
        count += 1
        
        get_job = jobs[job_id]
        occupation = get_job['occupation']
        skills = get_job['cross-skills']
        top10_skills = skills
        knowledge = get_job['knowledge']
        top10_knowledge = knowledge

        if skills:
            top10_skills = skills[:10]

        if knowledge:
            top10_knowledge = knowledge[:10]

        result = {
            'Score': score,
            'Job Title': occupation,
            'Top 10 Skills': top10_skills,
            'Top 10 Knowledge': top10_knowledge,
        }

        result = json.dumps(result, default=np_encoder)
        output.append(result)
        
    # print(output)
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