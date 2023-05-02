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
from helpers.OnetCsvHandler import OnetCsvHandler
from helpers import onet_recommender_tools as ort
from helpers import onet_downloader
from helpers.QueryHandler import QueryHandler

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

TEST = True

def get_data(download=True):
    raw_data_dir = os.path.join(os.environ['ROOT_PATH'], "backend/helpers/raw_data/")
    
    if download:
        onet_downloader.mkdir(raw_data_dir)
        
        onet_downloader.download_interests(raw_data_dir)
        onet_downloader.download_knowledge(raw_data_dir)
        onet_downloader.download_values(raw_data_dir)
        onet_downloader.download_cross_skills(raw_data_dir)

    csv_handler = OnetCsvHandler()
    csv_handler.generate_onet_dictionary(raw_data_dir)
    return csv_handler.data()

jobs = get_data(False if TEST else True)
inv_idx = ort.inverted_index(jobs)
job_idx_map = ort.job_to_idx(jobs)
skills_idx_map = ort.skill_to_idx(inv_idx)
n_docs = len(jobs.items())
idf = ort.compute_idf(inv_idx, n_docs)
doc_norms = ort.compute_doc_norms(inv_idx, idf, n_docs, job_idx_map)

skillsTrainingCsvPath = os.path.join(os.environ['ROOT_PATH'], "backend/helpers/querySkillsTraining.csv")
queryHandler = QueryHandler()
queryHandler.load_csv_training_data(skillsTrainingCsvPath)
queryHandler.train()
print(queryHandler.query("I like programming"))

@app.route("/")
def home():
    return render_template('careerfinder.html', title="sample html")

@app.route("/search")
def career_search():
    text = request.args.get("interest")
    result = ort.top10_results(text, jobs, inv_idx, idf, doc_norms, job_idx_map)
    print(result)
    return result

if TEST:
    app.run(debug=True)