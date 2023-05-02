import spacy
from spacy.training import Example
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
import pandas as pd
import random

class QueryHandler(object):

    def __init__(self):
        # Load the language model instance in spaCy
        self.nlp = spacy.load("en_core_web_sm")
        self.training_examples = []
                
    def load_csv_training_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.reset_index()
        skills = [skill.upper() for skill in df.loc[:,"Skill"]]

        for index, row in df.iterrows():
            sentences = row['Corpus'].split(".")
            row_skill = row['Skill'].upper()
            
            for doc in self.nlp.pipe(sentences):
                cats = {skill:False for skill in skills}
                cats[row_skill] = True
                example = Example.from_dict(doc, {"cats":cats})
                self.training_examples.append(example)

    def train(self):
        config = {
        "threshold": 0.5,
        "model": DEFAULT_MULTI_TEXTCAT_MODEL,
        }
        self.textcat = self.nlp.add_pipe("textcat_multilabel", config=config)
        self.textcat.initialize(lambda:self.training_examples, nlp=self.nlp)


    def query(self, text):
        doc = self.nlp(text)
        return sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
