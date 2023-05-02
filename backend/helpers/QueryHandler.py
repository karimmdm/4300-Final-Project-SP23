import spacy
from spacy.training import Example
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
import pandas as pd
import gensim.downloader


class QueryHandler(object):

    def __init__(self, use_gensim=True):
        # Load the language model instance in spaCy
        self.use_gensim = use_gensim
        if use_gensim:
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
        self.nlp = spacy.load("en_core_web_sm")
        self.training_examples = []
        self.skills = []
        
                
    def load_csv_training_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.reset_index()
        self.skills = [skill.lower() for skill in df.loc[:,"Skill"]]
        print(self.skills)
        for index, row in df.iterrows():
            sentences = row['Corpus'].split(".")
            row_skill = row['Skill'].upper()
            
            for doc in self.nlp.pipe(sentences):
                cats = {skill:False for skill in self.skills}
                cats[row_skill] = True
                example = Example.from_dict(doc, {"cats":cats})
                self.training_examples.append(example)

    def trainCNN(self):
        config = {
        "threshold": 0.5,
        "model": DEFAULT_MULTI_TEXTCAT_MODEL,
        }
        self.textcat = self.nlp.add_pipe("textcat_multilabel", config=config)
        self.textcat.initialize(lambda:self.training_examples, nlp=self.nlp)


    def query(self, text):
        doc = self.nlp(text)
        print(doc)
        valid_tokens = []
        for token in doc:
            if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADV', 'ADJ']:
                valid_tokens.append(token.text)
        valid_text = ""
        for token in valid_tokens:
            if token in self.skills:
                valid_text += token + ";"
        
        similar_words = []
        if self.use_gensim:
            similar_words = [self.glove_vectors.most_similar(token) for token in valid_tokens]
            print("VALID_SIMILAR", similar_words)
       
        for entry in similar_words:
            for word, score in entry:
                if word in self.skills:
                    valid_text += word + ";"
        if(valid_text != ""):
            return valid_text
        
        scores = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
        txt = ""
        for skill, score in scores:
            if score >= .80:
                txt += (skill.lower() + ";")
        return txt
