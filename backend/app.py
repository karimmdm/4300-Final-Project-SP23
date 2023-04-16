import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from collections import defaultdict
from collections import Counter
import json
import math
import numpy as np
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from IPython.core.display import HTML
import sys
from helpers import  OnetCsvHandler
import recommender_tools as rt
from helpers import onet_downloader

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
    raw_data_dir = os.path.join(os.environ['ROOT_PATH'], "backend/helpers/raw_data")
    onet_downloader.mkdir(raw_data_dir)
    
    onet_downloader.download_interests(raw_data_dir)
    onet_downloader.download_knowledge(raw_data_dir)
    onet_downloader.download_values(raw_data_dir)
    onet_downloader.download_cross_skills(raw_data_dir)

    csv_handler = OnetCsvHandler.OnetCsvHandler()
    csv_handler.generate_onet_dictionary(raw_data_dir)
    return csv_handler.data()

jobs = get_data()
inv_idx = rt.inverted_index(jobs)
job_idx_map = rt.job_to_idx(jobs)
skills_idx_map = rt.skill_to_idx(inv_idx)
n_docs = len(jobs.items())
idf = rt.compute_idf(inv_idx, n_docs)
doc_norms = rt.compute_doc_norms(inv_idx, idf, n_docs)


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/search")
def career_search():
    text = request.args.get("interest")
    return rt.top10_results(text, jobs, inv_idx, idf, doc_norms)


# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
# def sql_search(episode):
#     query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#     keys = ["id","title","descr"]
#     data = mysql_engine.query_selector(query_sql)
#     return json.dumps([dict(zip(keys,i)) for i in data])

app.run(debug=True)