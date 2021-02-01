#!/usr/bin/env python
# coding: utf-8

# ## EC Analysis -  A/B Class
# - Data Preprocessing
# - Focus on each product class
# - Analyzing Rücksprung Kommentar - frequent words (unigram/bigram/trigram)
# - Analyzing influencing factors of processing time of Bündels
# - Change Impact Analysis: predicting affected components of new Bündel
# - Predicting processing time of new Bündel

# In[574]:


# import libraries

import pandas as pd
import numpy as np
import scipy
from scipy import spatial
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import utils as skl_utils
from tqdm import tqdm
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler  
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer


from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from skmultilearn.ensemble import MajorityVotingClassifier, LabelSpacePartitioningClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

import multiprocessing

import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, GridSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns #visualisation
sns.set(color_codes=True)


# ## 1. Data Preprocessing - Load Data into Dataframes

# In[575]:


# load EC releted entities
# EC_Entity, EC_Kategorie, Kategorie, Modulekomponente

home_dir_tables = "/home/yuwepan/Promotion/Data/cmm2.0/"
# load EC Entity
df_ec_entity = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD101_ECEntity.parquet', engine='pyarrow')
df_ec_entity = df_ec_entity[['ID','EC_NUMMER','STATUS','STICHWORT','IST_ZUSTAND',
                            'SOLL_ZUSTAND','MODULKOMPONENTE','VERURSACHER','VERANTWORTLICHER_FK']]
df_ec_entity.columns = ['EC_ID','EC_NUMMER','EC_STATUS','EC_STICHWORT','EC_IST_ZUSTAND','EC_SOLL_ZUSTAND',
                        'EC_MK','EC_VERURSACHER','EC_VERANTWORTLICHER_FK']

# load EC kategorie 
df_ec_kategorie = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD107_KategorieZumECEntity.parquet', engine='pyarrow')
df_ec_kategorie = df_ec_kategorie[['EC_FK','KATEGORIE']]
df_ec_kategorie.columns = ['EC_FK','EC_KATEGORIE']

# load kategorie
df_kategorie = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD101_KAT.parquet', engine='pyarrow')
df_kategorie.columns = ['KATEGORIE','KAT_BENENNUNG']

# load mk
df_mk = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD101_MK.parquet', engine='pyarrow')
df_mk.columns = ['MK','MK_BENENNUNG']

# merge EC_entity with EC kategorie
df_ec = df_ec_entity.merge(df_ec_kategorie,left_on='EC_ID',right_on='EC_FK')
df_ec = df_ec.merge(df_kategorie,left_on='EC_KATEGORIE',right_on='KATEGORIE')

# merge EC with mk to get mk benennung
df_ec = df_ec.merge(df_mk,left_on='EC_MK',right_on='MK')


# In[576]:


df_ec.head()


# In[577]:


# load buendel related entities
# Buendel Entity, Planeinsatztermin, Paket, BR_AA, BuendelZustand, 
df_buendel_entity = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD151_BuendelEntity.parquet', engine='pyarrow')
df_buendel_entity = df_buendel_entity[['ID','CREATION_DATE','BUENDEL_NUMMER','STATUS','BENENNUNG',
                                       'GEWUENSCHTER_EINSATZ','KOSTEN_RELEVANT', 'ZERTIFIZIERUNGS_RELEVANT',
                                       'PROZESS_STATUS','EC_FK','VERANTWORTLICHER_FK','GREMIUM_ENTSCHEIDUNG_MANUELL',
                                       'STATUS_GENEHMIGUNG','RUECKSPRUNG_BEANTRAGT','RUECKSPRUNG_KOMMENTAR',
                                      'KOSTENBEWERTUNGS_ART','BEGR_RUECKSPRUNG','RUECKMELDE_DATUM',
                                      'MODULKOMPONENTE','BUENDEL_ZUSTAND_FK']]

df_buendel_entity.columns = ['Bnd_ID','Bnd_CREATION_DATE','BUENDEL_NUMMER','Bnd_STATUS','Bnd_BENENNUNG',
                                       'Bnd_GEWUENSCHTER_EINSATZ','Bnd_KOSTEN_RELEVANT', 'Bnd_ZERTIFIZIERUNGS_RELEVANT',
                                       'Bnd_PROZESS_STATUS','EC_FK','Bnd_VERANTWORTLICHER_FK','Bnd_GREMIUM_ENTSCHEIDUNG_MANUELL',
                                       'Bnd_STATUS_GENEHMIGUNG','Bnd_RUECKSPRUNG_BEANTRAGT','Bnd_RUECKSPRUNG_KOMMENTAR',
                                       'Bnd_KOSTENBEWERTUNGS_ART','Bnd_BEGR_RUECKSPRUNG','Bnd_RUECKMELDE_DATUM',
                                       'Bnd_MODULKOMPONENTE','BUENDEL_ZUSTAND_FK']

# load buendel_zustand
df_buendel_zustand = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD167_BuendelZustandEntity.parquet', engine='pyarrow')
df_buendel_zustand = df_buendel_zustand[['ID','IST_ZUSTAND','SOLL_ZUSTAND']]

df_buendel_zustand.columns = ['BUENDEL_ZUSTAND_ID','Bnd_IST_ZUSTAND','Bnd_SOLL_ZUSTAND']

# load planeinsatztermin
df_pet_entity = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD181_PlaneinsatzTerminEntity.parquet', engine='pyarrow')
df_pet_entity = df_pet_entity[['ID','PAKET_FK','BUENDEL_FK']]

df_pet_entity.columns = ['PET_ID','PAKET_FK','BUENDEL_FK']

# load paket
df_paket = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRMDBM161_PaketEntity.parquet', engine='pyarrow')
df_paket = df_paket[['ID','BRAA_FK','BENENNUNG','STATUS']]

df_paket.columns = ['Paket_ID','BR_AA_FK','Paket_BENENNUNG','Paket_STATUS']

# load br aa
df_braa = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRMDBM152_AdminBRAAEntity.parquet', engine='pyarrow')
df_braa = df_braa[['ID','BR','AA','STATUS']]

df_braa.columns = ['BRAA_ID','BR','AA','BRAA_STATUS']

# merge
df_pet_paket = df_pet_entity.merge(df_paket, left_on='PAKET_FK',right_on='Paket_ID')
df_pet_paket_braa = df_pet_paket.merge(df_braa,left_on='BR_AA_FK',right_on='BRAA_ID')
new_df_bnd_entity = df_pet_paket_braa.merge(df_buendel_entity,left_on='BUENDEL_FK', right_on='Bnd_ID',
                                    how='left')
new_df_bnd_entity = new_df_bnd_entity.merge(df_buendel_zustand,left_on='BUENDEL_ZUSTAND_FK',right_on='BUENDEL_ZUSTAND_ID')
new_df_bnd_entity = new_df_bnd_entity.merge(df_ec, left_on='EC_FK', right_on='EC_ID')

new_df_bnd_entity = new_df_bnd_entity.merge(df_mk,left_on='Bnd_MODULKOMPONENTE',right_on='MK')


# In[578]:


new_df_bnd_entity.columns


# In[579]:


new_df_bnd_drop = new_df_bnd_entity [['EC_NUMMER', 'EC_STATUS', 'EC_STICHWORT',
       'EC_IST_ZUSTAND', 'EC_SOLL_ZUSTAND', 'EC_VERURSACHER',
       'EC_VERANTWORTLICHER_FK','EC_KATEGORIE',
       'KAT_BENENNUNG', 'Bnd_ID', 'Bnd_CREATION_DATE', 'BUENDEL_NUMMER', 'Bnd_STATUS',
       'Bnd_BENENNUNG', 'Bnd_GEWUENSCHTER_EINSATZ', 'Bnd_KOSTEN_RELEVANT',
       'Bnd_ZERTIFIZIERUNGS_RELEVANT', 'Bnd_PROZESS_STATUS', 'EC_FK_x',
       'Bnd_VERANTWORTLICHER_FK', 'Bnd_GREMIUM_ENTSCHEIDUNG_MANUELL',
       'Bnd_STATUS_GENEHMIGUNG', 'Bnd_RUECKSPRUNG_BEANTRAGT',
       'Bnd_RUECKSPRUNG_KOMMENTAR', 'Bnd_KOSTENBEWERTUNGS_ART',
       'Bnd_BEGR_RUECKSPRUNG', 'Bnd_RUECKMELDE_DATUM', 'Bnd_MODULKOMPONENTE','MK_BENENNUNG_y',
       'Bnd_IST_ZUSTAND','Bnd_SOLL_ZUSTAND','BR','AA']].drop_duplicates(subset=None, keep='first')


# In[580]:


new_df_bnd_drop.head(5)


# In[581]:


new_df_bnd_drop['Bnd_KOSTEN_RELEVANT'] = new_df_bnd_drop['Bnd_KOSTEN_RELEVANT'].fillna('NaN')
new_df_bnd_drop['Bnd_ZERTIFIZIERUNGS_RELEVANT'] = new_df_bnd_drop['Bnd_ZERTIFIZIERUNGS_RELEVANT'].fillna('NaN')

new_df_bnd_drop['EC_STICHWORT'] = new_df_bnd_drop['EC_STICHWORT'].fillna('NaN')
new_df_bnd_drop['EC_IST_ZUSTAND'] = new_df_bnd_drop['EC_IST_ZUSTAND'].fillna('NaN')
new_df_bnd_drop['EC_SOLL_ZUSTAND'] = new_df_bnd_drop['EC_SOLL_ZUSTAND'].fillna('NaN')

new_df_bnd_drop['Bnd_BENENNUNG'] = new_df_bnd_drop['Bnd_BENENNUNG'].fillna('NaN')
new_df_bnd_drop['Bnd_RUECKSPRUNG_KOMMENTAR'] = new_df_bnd_drop['Bnd_RUECKSPRUNG_KOMMENTAR'].fillna('NaN')
new_df_bnd_drop['Bnd_IST_ZUSTAND'] = new_df_bnd_drop['Bnd_IST_ZUSTAND'].fillna('NaN')
new_df_bnd_drop['Bnd_SOLL_ZUSTAND'] = new_df_bnd_drop['Bnd_SOLL_ZUSTAND'].fillna('NaN')


# In[582]:


new_df_bnd_drop.loc[new_df_bnd_drop['Bnd_IST_ZUSTAND'].str.contains('siehe EC')]['BUENDEL_NUMMER'].nunique()


# In[583]:


new_df_bnd_drop.loc[new_df_bnd_drop['EC_IST_ZUSTAND'].str.contains('siehe die Beschreibung des Bündels')]['BUENDEL_NUMMER'].nunique()


# In[584]:


new_df_bnd_drop['BUENDEL_NUMMER'].nunique()


# In[585]:


# load produkt, convert BR to product class
df_produkt = pd.read_csv('/home/yuwepan/Promotion/Data/Produkt.csv',sep="|",header=0)


# In[586]:


# convert unknown product class 
def convert_pk(p):
    tmp_p = ""
    if p == 'unbekannt C223 / FVV' :
        tmp_p = 'S- SL-Klasse'
    elif p == 'unbekannt C212 / BZ3':
        tmp_p = 'E-Klasse'
    else:
        tmp_p = p
    return tmp_p
        


# In[587]:


df_produkt['new_ProduktKlasse'] = df_produkt['ProduktKlasse'].apply(lambda x: convert_pk(x))


# In[588]:


df_produkt


# In[589]:


df_produkt.loc[df_produkt['ProduktKlasse'].str.contains('C212')]


# In[590]:


df_produkt = df_produkt[['BR','new_ProduktKlasse']].drop_duplicates(subset=None, keep='first')
df_produkt


# In[591]:


df_produkt.loc[df_produkt['BR']=='C177']


# In[592]:


# add column 'new_ProduktKlase' to merged buendel entity
new_df_bnd_drop_produkt = new_df_bnd_drop.merge(df_produkt,left_on='BR',right_on='BR')


# In[593]:


new_df_bnd_drop_produkt['new_ProduktKlasse'].nunique()


# In[594]:


new_df_bnd_drop_produkt.groupby(['BR'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[595]:


new_df_bnd_drop_produkt.groupby(['new_ProduktKlasse'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[596]:


new_df_bnd_drop_produkt['BUENDEL_NUMMER'].nunique()


# In[597]:


new_df_bnd_drop_produkt.head(3)


# In[598]:


def new_IST(df):
    if df['Bnd_IST_ZUSTAND'] == 'siehe EC':
        val = df['EC_IST_ZUSTAND']
    else:
        val = df['Bnd_IST_ZUSTAND']
    return val

def new_SOLL(df):
    if df['Bnd_SOLL_ZUSTAND'] == 'siehe EC':
        val = df['EC_SOLL_ZUSTAND']
    else:
        val = df['Bnd_SOLL_ZUSTAND']
    return val


# ## 2. Focus on Bündels in each product class
# - A/B Class

# In[599]:


#new_df_bnd_produkt_s = new_df_bnd_drop_produkt.loc[new_df_bnd_drop_produkt['new_ProduktKlasse'].str.contains('A- B-Klasse')]
#new_df_bnd_produkt_s = new_df_bnd_drop_produkt.loc[new_df_bnd_drop_produkt['new_ProduktKlasse'].str.contains('S- SL-Klasse')]
new_df_bnd_produkt_s = new_df_bnd_drop_produkt.loc[new_df_bnd_drop_produkt['new_ProduktKlasse'].str.contains('C-Klasse')]

new_df_bnd_produkt_s.head()


# In[600]:


new_df_bnd_produkt_s.nunique()


# In[601]:


# for bundels which have no IST/SOLl Zustand, use the IST/SOLl Zustand from EC
new_df_bnd_produkt_s['new_IST'] = new_df_bnd_produkt_s.apply(new_IST, axis=1)
new_df_bnd_produkt_s['new_SOLL'] = new_df_bnd_produkt_s.apply(new_SOLL, axis=1)


# In[602]:


new_df_bnd_produkt_s.head()


# In[603]:


new_df_bnd_produkt_s.columns


# In[604]:


# choose relevant columns to build Bündel Dataframe
df_bnd_s = new_df_bnd_produkt_s[['Bnd_ID', 'BUENDEL_NUMMER','Bnd_CREATION_DATE','Bnd_RUECKMELDE_DATUM',
                                 'Bnd_STATUS','Bnd_BENENNUNG','EC_STICHWORT','EC_VERURSACHER', 'KAT_BENENNUNG','new_IST','new_SOLL',
                                 'Bnd_GEWUENSCHTER_EINSATZ','Bnd_KOSTEN_RELEVANT','Bnd_ZERTIFIZIERUNGS_RELEVANT',
                                 'Bnd_PROZESS_STATUS','Bnd_VERANTWORTLICHER_FK','Bnd_GREMIUM_ENTSCHEIDUNG_MANUELL',
                                 'Bnd_STATUS_GENEHMIGUNG','Bnd_RUECKSPRUNG_KOMMENTAR','Bnd_KOSTENBEWERTUNGS_ART',
                                 'Bnd_BEGR_RUECKSPRUNG','Bnd_MODULKOMPONENTE','MK_BENENNUNG_y','new_ProduktKlasse']].drop_duplicates(subset=None, keep='first')


# In[605]:


df_bnd_s.columns = ['Bnd_ID', 'BUENDEL_NUMMER', 'CREATION_DATE', 'RUECKMELDE_DATUM',
                    'STATUS', 'BENENNUNG', 'STICHWORT','VERURSACHER','KAT', 'new_IST','new_SOLL', 'GEWUENSCHTER_EINSATZ',
                    'KOSTEN_RELEVANT','ZERTIFIZIERUNGS_RELEVANT','PROZESS_STATUS',
                    'VERANTWORTLICHER_FK', 'GREMIUM_ENTSCHEIDUNG_MANUELL', 'STATUS_GENEHMIGUNG',
                    'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART', 'BEGR_RUECKSPRUNG', 
                    'MK', 'MK_BENENNUNG', 'ProduktKlasse']


# In[606]:


df_bnd_s.head()


# In[607]:


df_bnd_s['ZERTIFIZIERUNGS_RELEVANT'].unique()


# In[608]:


df_bnd_s.groupby(['KOSTENBEWERTUNGS_ART'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# ### Remove test ECs

# In[609]:


# test ECs
indexNames_testecs = df_bnd_s.loc[(df_bnd_s['BENENNUNG'].isin(['test','Test'])) 
                   | (df_bnd_s['STICHWORT'].isin(['test','Test'])) 
                   | (df_bnd_s['new_IST'].isin(['test','Test','Siehe Anhang','siehe jeweiliges Bündel']))
                   | (df_bnd_s['new_SOLL'].isin(['test','Test','Siehe Anhang','siehe jeweiliges Bündel']))].index.tolist()
indexNames_testecs


# In[610]:


df_bnd_s.drop(indexNames_testecs,inplace=True)


# In[611]:


df_bnd_s.head()


# In[612]:


# Count word length in each description
df_bnd_s['word_count_benennung'] = df_bnd_s['BENENNUNG'].apply(lambda x: len(str(x).split(" ")))
df_bnd_s['word_count_stichwort'] = df_bnd_s['STICHWORT'].apply(lambda x: len(str(x).split(" ")))
df_bnd_s['word_count_rueck'] = df_bnd_s['RUECKSPRUNG_KOMMENTAR'].apply(lambda x: len(str(x).split(" ")))
df_bnd_s['word_count_berg_rueck'] = df_bnd_s['BEGR_RUECKSPRUNG'].apply(lambda x: len(str(x).split(" ")))
df_bnd_s['word_count_ist'] = df_bnd_s['new_IST'].apply(lambda x: len(str(x).split(" ")))
df_bnd_s['word_count_soll'] = df_bnd_s['new_SOLL'].apply(lambda x: len(str(x).split(" ")))


# In[613]:


df_bnd_s


# In[614]:


df_bnd_s.groupby(['word_count_ist'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=True).head(10)


# In[615]:


df_bnd_s['word_count_stichwort'].describe()


# In[616]:


df_buendel_kat_s=df_bnd_s[['BUENDEL_NUMMER','KAT']].drop_duplicates()
df_buendel_kat_s['KAT'] = df_bnd_s['KAT'].astype('str')
concat_df_buendel_kat_s = df_bnd_s.groupby(['BUENDEL_NUMMER'])['KAT'].apply(','.join).reset_index()

# sort strings in column EC_KATEGORIE of df_x
for i, row in concat_df_buendel_kat_s.iterrows():
    tmp_cat = list(row['KAT'].split(sep=','))
    tmp_cat.sort()
    cat = ','.join([str(elem) for elem in tmp_cat]) 
    #temp_art = ' '.join(set(row['ProduktArt'].split(sep=',')))
    concat_df_buendel_kat_s.at[i,'KAT'] = cat

concat_df_buendel_kat_s


# In[617]:


df_bnd_s = df_bnd_s.merge(concat_df_buendel_kat_s, left_on='BUENDEL_NUMMER', right_on='BUENDEL_NUMMER')
df_bnd_s.head()


# In[618]:


#df_bnd_s = df_bnd_s.drop(columns=['KAT_x'])
df_bnd_s.rename(columns={'KAT_x':'KAT'})
df_bnd_s.drop(columns=['KAT_y'])


# In[619]:


df_bnd_s = df_bnd_s.drop_duplicates(subset=None, keep='first')
df_bnd_s.head()


# ## 3. Analyzing Rücksprung Kommentar - frequent words (unigram/bigram/trigram)

# In[620]:


df_bnd_s.loc[df_bnd_s['RUECKSPRUNG_KOMMENTAR'].str.contains('NaN')==False]['BUENDEL_NUMMER'].nunique()


# ### Preprocessing 

# In[621]:


list_ruck_str = []
list_ruck_str = df_bnd_s.loc[df_bnd_s['RUECKSPRUNG_KOMMENTAR'].str.contains('NaN')==False]['RUECKSPRUNG_KOMMENTAR'].values.tolist()
list_ruck_str


# In[622]:


df_bnd_s.loc[df_bnd_s['RUECKSPRUNG_KOMMENTAR'].str.contains('Kostenbewertung neu beauftragen')]['BUENDEL_NUMMER']


# In[623]:


len(list_ruck_str)


# In[624]:


from nltk.stem.snowball import GermanStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = SnowballStemmer('german')

stops = set(stopwords.words('german'))
stops.update(set(stopwords.words('english')))

stops.update(['bündel','wurde','wurden','aufgrund','hinzu','bitte','wegen','siehe','rücksprung','bereits'])
  

def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    
    for idx in range(len(docs)):
        if docs[idx]:
            #docs[idx] = str(docs[idx])
            #docs[idx] = docs[idx].split('<E>', 1)[0] # Delete English tranlation, first focus on German text
        
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc or [] if not token.isdigit()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc or [] if len(token) > 2] for doc in docs]
    
    #lemmatizer = WordNetLemmatizer()
    #docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    
    # Remove stopwords
    docs = [[token for token in doc or [] if not token in stops] for doc in docs]
    
    stemmer = GermanStemmer()
    docs = [[stemmer.stem(token) for token in doc] for doc in docs]
    
    # Lemmatize all words in documents.
    # For German, import snowball
    #lemmatizer = WordNetLemmatizer()
    #docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    #stemmer = GermanStemmer()
    #docs = [[stemmer.stem(token) for token in doc] for doc in docs]
    return docs


# In[625]:


docs = []
docs = docs_preprocessor(list_ruck_str)
docs


# ### Most frequent words in Rucksprung Kommentar

# In[626]:


from nltk.probability import FreqDist
# fdist = FreqDist()

# for idx in range(len(list_str)):
#     if list_str[idx]:
#         words= nltk.word_tokenize(list_str[idx])
#         for word in words:
#             fdist[word] += 1

fdist = FreqDist()
for idx in range(len(docs)):
    if docs[idx]:
        #for word in range(len(docs[idx])):
        for word in docs[idx]:
            fdist[word] += 1


# In[627]:


fdist.N()


# In[628]:


fdist.most_common(50)


# ### Creating a vector of word counts 

# In[629]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_df=0.8, stop_words=stops, max_features=10000, ngram_range=(1,3))


# In[630]:


import string
str_docs = []
for i in range(len(docs)):
    str_docs.append([])   # appending a new list!

for idx in range(len(docs)):
    str_docs[idx] = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in docs[idx]]).strip()

str_docs[1]


# In[631]:


X = cv.fit_transform(str_docs)


# In[632]:


list(cv.vocabulary_.keys())[:10]


# In[633]:


#Most frequently occuring words
def get_top_n_words(docs, n=None):
    vec = CountVectorizer().fit(docs)
    bag_of_words = vec.transform(docs)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]


# In[634]:


#Convert most freq words to datagrame for plotting bar plot
top_words = get_top_n_words(str_docs, n=30)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word","Freq"]

#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(20,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)


# In[635]:


#Most frequently occuring Bi-grams
def get_top_n2_words(docs, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(docs)
    bag_of_words = vec1.transform(docs)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

top2_words = get_top_n2_words(str_docs, n=30)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)

#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(25,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=90)


# In[636]:


#Most frequently occuring Tri-grams
def get_top_n3_words(docs, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(docs)
    bag_of_words = vec1.transform(docs)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

top3_words = get_top_n3_words(str_docs, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)

#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(20,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=90)


# In[637]:


index_ruck_snr = df_bnd_s.loc[df_bnd_s['RUECKSPRUNG_KOMMENTAR'].str.contains('SNR')]['RUECKSPRUNG_KOMMENTAR'].index.tolist()
index_ruck_sachnummer = df_bnd_s.loc[df_bnd_s['RUECKSPRUNG_KOMMENTAR'].str.contains('Sachnummer')]['RUECKSPRUNG_KOMMENTAR'].index.tolist()
index_ruck_teil = df_bnd_s.loc[df_bnd_s['RUECKSPRUNG_KOMMENTAR'].str.contains('Teil')]['RUECKSPRUNG_KOMMENTAR'].index.tolist()


# In[638]:


list_ruck_snr_relevant = []
list_ruck_snr_relevant = list(set(index_ruck_snr + index_ruck_sachnummer + index_ruck_teil))
len(list_ruck_snr_relevant)


# ## 4. Analyzing influencing factors of processing time of Bündels

# In[639]:


df_bnd_s.columns


# In[640]:


# Focus on Abgeschlossen EC
df_bnd_s_abgeschlossen = df_bnd_s.loc[df_bnd_s['PROZESS_STATUS'] == 'ABGESCHLOSSEN']


# In[641]:


df_bnd_s_abgeschlossen['BUENDEL_NUMMER'].nunique()


# In[642]:


df_bnd_s['BUENDEL_NUMMER'].nunique()


# In[643]:


# for S-Class, there are 24462 Bündels
# Focus on 'Abgeschlossen' Bündels
df_bnd_s.groupby(['PROZESS_STATUS'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[644]:


df_bnd_s_abgeschlossen.groupby(['RUECKMELDE_DATUM'])['BUENDEL_NUMMER'].nunique()


# In[645]:


lisst_ruckdatum = df_bnd_s_abgeschlossen['RUECKMELDE_DATUM'].tolist()


# In[646]:


lisst_ruckdatum.sort()


# In[647]:


# Feedback date is from 2013 to 2020
lisst_ruckdatum


# In[648]:


# Calculate duration of processing time of finished ECs
df_bnd_s_abgeschlossen['CREATION_DATE'] = df_bnd_s_abgeschlossen['CREATION_DATE'].apply(lambda x: pd.to_datetime(x).date())
df_bnd_s_abgeschlossen['RUECKMELDE_DATUM'] = df_bnd_s_abgeschlossen['RUECKMELDE_DATUM'].apply(lambda x: pd.to_datetime(x).date())

df_bnd_s_abgeschlossen['DAUER'] = df_bnd_s_abgeschlossen['RUECKMELDE_DATUM'] - df_bnd_s_abgeschlossen['CREATION_DATE']


# In[649]:


df_bnd_s_abgeschlossen[['CREATION_DATE','RUECKMELDE_DATUM','DAUER']]


# In[650]:


# drop bundels with wrong Ruckmeldung_datum, e.g. in year 2033...2099 
df_bnd_s_abgeschlossen_dauer = df_bnd_s_abgeschlossen.loc[(df_bnd_s_abgeschlossen['DAUER'] > '1000 days')==False]
df_bnd_s_abgeschlossen_dauer


# In[651]:


# average duration of Bundel processing time for S-Class is 66 days
df_bnd_s_abgeschlossen_dauer['DAUER'].describe()


# In[652]:


# Histogram Dauer
quantile_list = [0, .25, .5, .75, 1.]


quantiles = df_bnd_s_abgeschlossen_dauer['DAUER'].quantile(quantile_list).astype('timedelta64[D]').quantile(quantile_list)
quantiles

# quantiles_new = df_bnd_s_abgeschlossen_dauer.loc[input_df_cal_dauer['RUECKSPRUNG_KOMMENTAR'].str.contains('nan')==False]['Dauer'].astype('timedelta64[D]').quantile(quantile_list)
# quantiles_new


# In[653]:


fig, ax = plt.subplots()
df_bnd_s_abgeschlossen_dauer['DAUER'].astype('timedelta64[D]').hist(bins=30, color='#A9C5D3', 
                             edgecolor='black', grid=False)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)
ax.set_title('Dauer von Änderungsanträgen bis zur Rückmeldung "Produktiv abgeschlossen"', 
             fontsize=12)
ax.set_xlabel('Dauer in Kalendertagen', fontsize=12)
ax.set_ylabel('Häufigkeit', fontsize=12)


# In[654]:


# Labeling Bündel with Duration Category
# def dauer_cat(row):
#     if row['DAUER'] > pd.to_timedelta('42 days'):
#         val = 'longer'
#     elif row['DAUER'] < pd.to_timedelta('42 days'):
#         val = 'shorter'
#     else:
#         val = 'equal'
#     return val

def dauer_cat(row):
    if row['DAUER'] <= pd.to_timedelta('43 days'):
        val = 'Shorter'
    #elif (row['DAUER'] > pd.to_timedelta('42 days')) & (row['DAUER'] <= pd.to_timedelta('64 days')):
        #val = 'Q2'
#     elif (row['DAUER'] > pd.to_timedelta('33 days')) & (row['DAUER'] <= pd.to_timedelta('64 days')):
#         val = 'Q3'
    else:
        val = 'Longer'
    return val

df_bnd_s_abgeschlossen_dauer['DAUER_KAT'] = df_bnd_s_abgeschlossen_dauer.apply(dauer_cat, axis=1)


# In[655]:


df_bnd_s_abgeschlossen_dauer[['DAUER','DAUER_KAT']]


# ## 5. Change Impact Analysis

# ### Load SNR related data

# In[656]:


df_snrimbuendel = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD301_SachnummerImBuendelEntity.parquet', engine='pyarrow')


# In[657]:


df_snrimbuendel = df_snrimbuendel[['SACHNUMMER','BUENDEL_FK','MODULSUBKOMPONENTE','ZB_BENENNUNG','MODULGRUPPE']]


# In[658]:


# merge with buendel entity to get buendel number
new_df_linkbuendelsnr = df_bnd_s_abgeschlossen_dauer.merge(df_snrimbuendel, left_on='Bnd_ID', right_on='BUENDEL_FK')


# In[659]:


# choose relavent columns to form a new dataframe
df_linkbuendelsnr = pd.DataFrame
df_linkbuendelsnr = new_df_linkbuendelsnr[['BUENDEL_NUMMER','SACHNUMMER','MODULSUBKOMPONENTE','ZB_BENENNUNG','MODULGRUPPE']]
df_linkbuendelsnr


# In[660]:


# only focus on A-SNR
df_linkbuendelsnr['Kennbuchstabe'] = df_linkbuendelsnr['SACHNUMMER'].astype(str).str[0]
indexNames_dropsnr_notA = df_linkbuendelsnr.loc[df_linkbuendelsnr['Kennbuchstabe'] != 'A'].index
df_linkbuendelsnr.drop(indexNames_dropsnr_notA, inplace=True)


# In[661]:


df_linkbuendelsnr['MODULSUBKOMPONENTE'] = df_linkbuendelsnr['MODULSUBKOMPONENTE'].fillna(0)
df_linkbuendelsnr['MODULGRUPPE'] = df_linkbuendelsnr['MODULGRUPPE'].fillna(999)

# convert columns to appropriate datatype
df_linkbuendelsnr['BUENDEL_NUMMER'] = df_linkbuendelsnr['BUENDEL_NUMMER'].astype('str')
df_linkbuendelsnr['SACHNUMMER'] = df_linkbuendelsnr['SACHNUMMER'].astype('str')
df_linkbuendelsnr['MODULSUBKOMPONENTE'] = df_linkbuendelsnr['MODULSUBKOMPONENTE'].astype('int')
df_linkbuendelsnr['ZB_BENENNUNG'] = df_linkbuendelsnr['ZB_BENENNUNG'].astype('str')
df_linkbuendelsnr['MODULGRUPPE'] = df_linkbuendelsnr['MODULGRUPPE'].astype('int')


# In[662]:


# extract MK und MSK from the 5-digit MODULSUBKOMPONENT
for i, row in df_linkbuendelsnr.iterrows():
    temp_msk = str(row['MODULSUBKOMPONENTE']).zfill(5)
    #df_linkbuendelsnr.set_value(i,'MSK',temp_msk)
    df_linkbuendelsnr.at[i,'MSK'] = temp_msk


# In[663]:


for i, row in df_linkbuendelsnr.iterrows():
    temp_mk = row['MSK'][0:3]
    temp_msubk = row['MSK'][3:5]
    #df_linkbuendelsnr.set_value(i,'MK',temp_mk)
    #df_linkbuendelsnr.set_value(i,'SUBK',temp_msubk)
    df_linkbuendelsnr.at[i,'MK'] = temp_mk
    df_linkbuendelsnr.at[i,'SUBK'] = temp_msubk


# In[664]:


# drop items in which MG or MK is invalid
df_linkbuendelsnr = df_linkbuendelsnr.drop('MODULSUBKOMPONENTE', 1)


# ### Decompose SNR into:
# 
# * Kennbuchstabe (SNR-Kennbuchstabe) → 1. Stelle in der A-Sachnummer
# * Typzahl → 2. bis einschließlich 4. Stelle in der A-Sachnummer
# * Konstruktions_Haupt_und_Untergruppe (Konstruktions-Haupt- und Untergruppe) → 5. bis einschließlich 7. Stelle der A-Sachnummer
# * Fortlaufende_Nummer (Abwandlung oder fortlaufende Nummer) → 8. und 9. Stelle der A-Sachnummer
# * Teilnummer_Untergruppe (Teilnummer bezogen auf die Untergruppe) → 10. und 11. Stelle der A-Sachnummer

# In[665]:


for i, row in df_linkbuendelsnr.iterrows():
    temp_typzahl = 'No Value'
    temp_kg = 'No Value'
    temp_u = 'No Value'
    temp_kgu = 'No Value'
    temp_fortlaufende_nummer = 'No Value'
    temp_teilnummer_untergruppe = 'No Value'
    
    temp_typzahl = row['SACHNUMMER'][1:4]
    temp_kg = row['SACHNUMMER'][4:6]
    temp_u = row['SACHNUMMER'][6:7]
    temp_kgu = row['SACHNUMMER'][4:7]
    temp_fortlaufende_nummer = row['SACHNUMMER'][7:9]
    temp_teilnummer_untergruppe = row['SACHNUMMER'][9:11]
    
#     df_snrimbuendel.set_value(i,'TypZahl',temp_typzahl)
#     df_snrimbuendel.set_value(i,'Konstruktions_Haupt_und_Untergruppe',temp_k_haupt_untergruppe)
#     df_snrimbuendel.set_value(i,'Fortlaufende_Nummer',temp_fortlaufende_nummer)
#     df_snrimbuendel.set_value(i,'Teilnummer_Untergruppe',temp_teilnummer_untergruppe)
    df_linkbuendelsnr.at[i,'TypZahl'] = temp_typzahl
    df_linkbuendelsnr.at[i,'KG'] = temp_kg
    df_linkbuendelsnr.at[i,'U'] = temp_u
    df_linkbuendelsnr.at[i,'KGU'] = temp_kgu
    df_linkbuendelsnr.at[i,'Fortlaufende_Nummer'] = temp_fortlaufende_nummer
    df_linkbuendelsnr.at[i,'Teilnummer_Untergruppe'] = temp_teilnummer_untergruppe


# ### Merge two dataframes: df_linkbuendelsnr and df_bnd_s_abgeschlossen_dauer, link SNRs with change description

# In[666]:


df_linkbuendelsnr


# In[667]:


df_linkbuendelsnr['BUENDEL_NUMMER'].nunique()


# In[668]:


df_buendel_linked = df_linkbuendelsnr.merge(df_bnd_s_abgeschlossen_dauer, left_on = 'BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[669]:


df_buendel_linked = df_buendel_linked.drop_duplicates()


# In[670]:


df_buendel_linked.columns


# In[671]:


df_buendel_linked = df_buendel_linked.drop(columns=['KAT_x'])


# In[672]:


df_buendel_linked.columns = ['BUENDEL_NUMMER', 'SACHNUMMER', 'ZB_BENENNUNG', 'MODULGRUPPE',
       'Kennbuchstabe', 'MSK', 'MK', 'SUBK', 'TypZahl', 'KG', 'U', 'KGU',
       'Fortlaufende_Nummer', 'Teilnummer_Untergruppe', 'Bnd_ID',
       'CREATION_DATE', 'RUECKMELDE_DATUM', 'STATUS', 'BENENNUNG', 'STICHWORT',
        'VERURSACHER','new_IST', 'new_SOLL', 'GEWUENSCHTER_EINSATZ', 'KOSTEN_RELEVANT',
       'ZERTIFIZIERUNGS_RELEVANT', 'PROZESS_STATUS', 'VERANTWORTLICHER_FK',
       'GREMIUM_ENTSCHEIDUNG_MANUELL', 'STATUS_GENEHMIGUNG',
       'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART', 'BEGR_RUECKSPRUNG',
       'Bnd_MK', 'Bnd_MK_BENENNUNG', 'ProduktKlasse', 'word_count_benennung',
       'word_count_stichwort', 'word_count_rueck', 'word_count_berg_rueck',
       'word_count_ist', 'word_count_soll', 'Bnd_KAT', 'DAUER', 'DAUER_KAT']


# In[673]:


df_buendel_linked = df_buendel_linked.merge(df_mk, left_on='MK', right_on='MK')


# In[674]:


df_buendel_linked


# In[675]:


# add prefix for MK/KG for better understanding
MK_arr = []
KG_arr = []
MK_arr = ['MK_'+ s for s in df_buendel_linked['MK'].values.astype(str)]
KG_arr = ['KG_'+ s for s in df_buendel_linked['KG'].values.astype(str)]

df_buendel_linked['MK'] = MK_arr
df_buendel_linked['KG'] = KG_arr


# In[676]:


# add prefix for MSK/KGU for better understanding
MSK_arr = []
KGU_arr = []
MSK_arr = ['MSK_'+ s for s in df_buendel_linked['MSK'].values.astype(str)]
KGU_arr = ['KGU_'+ s for s in df_buendel_linked['KGU'].values.astype(str)]

df_buendel_linked['MSK'] = MSK_arr
df_buendel_linked['KGU'] = KGU_arr


# In[677]:


df_buendel_linked.columns


# #### delete instances that have less target variable (MK/KG <50)

# In[678]:


df_buendel_linked.groupby(['MK_BENENNUNG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=True).head(45)


# In[679]:


df_buendel_linked.groupby(['KG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=True).head(25)


# In[680]:


index_tooless_mk = []
index_tooless_mk = df_buendel_linked.groupby(['MK_BENENNUNG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=True).head(45).index.tolist()
index_tooless_mk


# In[681]:


list_index_tooless_mk = []
list_index_tooless_mk= df_buendel_linked.loc[df_buendel_linked['MK_BENENNUNG'].isin(index_tooless_mk)].index
list_index_tooless_mk


# In[682]:


df_buendel_linked.drop(list_index_tooless_mk,inplace=True)


# In[683]:


index_tooless_kg = []
index_tooless_kg = df_buendel_linked.groupby(['KG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=True).head(25).index.tolist()
index_tooless_kg


# In[684]:


list_index_tooless_kg = []
list_index_tooless_kg = df_buendel_linked.loc[df_buendel_linked['KG'].isin(index_tooless_kg)].index
list_index_tooless_kg


# In[685]:


df_buendel_linked.drop(list_index_tooless_kg,inplace=True)


# #### Combine multi variables into one.

# In[686]:


df_buendel_typ = df_buendel_linked[['BUENDEL_NUMMER','TypZahl']].drop_duplicates()
concat_df_buendel_typ = df_buendel_typ.groupby(['BUENDEL_NUMMER'])['TypZahl'].apply(';'.join).reset_index()
concat_df_buendel_typ


# In[687]:


df_buendel_mk = df_buendel_linked[['BUENDEL_NUMMER','MK']].drop_duplicates()
concat_df_buendel_mk = df_buendel_mk.groupby(['BUENDEL_NUMMER'])['MK'].apply(';'.join).reset_index()
concat_df_buendel_mk


# In[688]:


df_buendel_subk = df_buendel_linked[['BUENDEL_NUMMER','SUBK']].drop_duplicates()
concat_df_buendel_subk = df_buendel_subk.groupby(['BUENDEL_NUMMER'])['SUBK'].apply(';'.join).reset_index()
concat_df_buendel_subk


# In[689]:


df_buendel_msk = df_buendel_linked[['BUENDEL_NUMMER','MSK']].drop_duplicates()
concat_df_buendel_msk = df_buendel_msk.groupby(['BUENDEL_NUMMER'])['MSK'].apply(';'.join).reset_index()
concat_df_buendel_msk


# In[690]:


df_buendel_kat=df_buendel_linked[['BUENDEL_NUMMER','Bnd_KAT']].drop_duplicates()
df_buendel_kat['Bnd_KAT'] = df_buendel_kat['Bnd_KAT'].astype('str')
concat_df_buendel_kat = df_buendel_kat.groupby(['BUENDEL_NUMMER'])['Bnd_KAT'].apply(';'.join).reset_index()

# sort strings in column EC_KATEGORIE of df_x
for i, row in concat_df_buendel_kat.iterrows():
    tmp_cat = list(row['Bnd_KAT'].split(sep=';'))
    tmp_cat.sort()
    cat = ','.join([str(elem) for elem in tmp_cat]) 
    #temp_art = ' '.join(set(row['ProduktArt'].split(sep=',')))
    concat_df_buendel_kat.at[i,'Bnd_KAT'] = cat

concat_df_buendel_kat


# In[691]:


concat_df_buendel_kat['Bnd_KAT'].nunique()


# In[692]:


df_buendel_kg = df_buendel_linked[['BUENDEL_NUMMER','KG']].drop_duplicates()
df_buendel_kg['KG'] = df_buendel_kg['KG'].astype('str')
concat_df_buendel_kg = df_buendel_kg.groupby(['BUENDEL_NUMMER'])['KG'].apply(';'.join).reset_index()
concat_df_buendel_kg


# In[693]:


df_buendel_u = df_buendel_linked[['BUENDEL_NUMMER','U']].drop_duplicates()
df_buendel_u['U'] = df_buendel_u['U'].astype('str')
concat_df_buendel_u = df_buendel_u.groupby(['BUENDEL_NUMMER'])['U'].apply(';'.join).reset_index()
concat_df_buendel_u


# In[694]:


df_buendel_kgu = df_buendel_linked[['BUENDEL_NUMMER','KGU']].drop_duplicates()
df_buendel_kgu['KGU'] = df_buendel_kgu['KGU'].astype('str')
concat_df_buendel_kgu = df_buendel_kgu.groupby(['BUENDEL_NUMMER'])['KGU'].apply(';'.join).reset_index()
concat_df_buendel_kgu


# ### Merge all variables via bnd-nummer

# In[695]:


concat_df_buendel_typ_mk = pd.merge(concat_df_buendel_typ, concat_df_buendel_mk, 
                                   left_on='BUENDEL_NUMMER', right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_typ_mk


# In[696]:


concat_df_buendel_mk_subk = pd.merge(concat_df_buendel_typ_mk, concat_df_buendel_subk, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_subk


# In[697]:


concat_df_buendel_mk_subk_msk = pd.merge(concat_df_buendel_mk_subk, concat_df_buendel_msk, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_subk_msk


# In[698]:


concat_df_buendel_mk_cat = pd.merge(concat_df_buendel_mk_subk_msk,concat_df_buendel_kat, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_cat


# In[699]:


concat_df_buendel_mk_cat_kg = pd.merge(concat_df_buendel_mk_cat, concat_df_buendel_kg, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_cat_kg


# In[700]:


concat_df_buendel_mk_cat_kg_u = pd.merge(concat_df_buendel_mk_cat_kg, concat_df_buendel_u, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_cat_kg_u


# In[701]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_mk_cat_kg_u, concat_df_buendel_kgu, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_linked


# In[702]:


df_amount_snr = df_buendel_linked.groupby(['BUENDEL_NUMMER'])['SACHNUMMER'].nunique().reset_index()

df_amount_snr_benennung = df_buendel_linked.groupby(['BUENDEL_NUMMER'])['ZB_BENENNUNG'].nunique().reset_index()


# In[703]:


df_amount_snr.columns = ['BUENDEL_NUMMER','ANZAHL_SACHNUMMER']
df_amount_snr


# In[704]:


df_amount_snr_benennung.columns = ['BUENDEL_NUMMER','ANZAHL_ZB_BENENNUNG']
df_amount_snr_benennung


# In[705]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, df_amount_snr, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, df_amount_snr_benennung, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_linked


# ### Merge description with part info

# In[706]:


input_df = pd.merge(df_bnd_s_abgeschlossen_dauer, concat_df_buendel_linked, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')

input_df


# In[707]:


input_df = input_df.drop(columns=['KAT_x','KAT_y'])


# In[708]:


input_df['ProduktKlasse']


# In[709]:


input_df.columns


# In[710]:


input_df.columns = ['Bnd_ID', 'BUENDEL_NUMMER', 'CREATION_DATE', 'RUECKMELDE_DATUM',
       'STATUS', 'BENENNUNG', 'STICHWORT', 'VERURSACHER','new_IST', 'new_SOLL',
       'GEWUENSCHTER_EINSATZ', 'KOSTEN_RELEVANT', 'ZERTIFIZIERUNGS_RELEVANT',
       'PROZESS_STATUS', 'VERANTWORTLICHER_FK', 'GREMIUM_ENTSCHEIDUNG_MANUELL',
       'STATUS_GENEHMIGUNG', 'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART',
       'BEGR_RUECKSPRUNG', 'Bnd_MK', 'Bnd_MK_BENENNUNG', 'ProduktKlasse',
       'word_count_benennung', 'word_count_stichwort', 'word_count_rueck',
       'word_count_berg_rueck', 'word_count_ist', 'word_count_soll', 
       'DAUER', 'DAUER_KAT', 'TypZahl', 'MK', 'SUBK', 'MSK', 'Bnd_KAT', 'KG',
       'U', 'KGU', 'ANZAHL_SACHNUMMER', 'ANZAHL_ZB_BENENNUNG']


# In[712]:


input_df[['Bnd_MK','Bnd_MK_BENENNUNG','MK','KG']]


# ### Delete instances with duplicated change description
# based on 'new_IST' and 'MK' (suppose same 'new_IST' has also the same 'new_SOLL' and 'Stichwort'
# - delete duplicate data, otherwise in test data may overlap with train data
# - concat MK based on the same 'new_IST', just like above: merge MK on the same BuendelNummer

# In[713]:


input_df_final = input_df.drop_duplicates(subset=['BENENNUNG','STICHWORT','new_IST','new_SOLL'],keep='first')
input_df_final


# ### Text Preprocessing + NLP Pipeline

# In[714]:


get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud
import matplotlib.pyplot as plt 

stops = set(stopwords.words('german'))
stops.update(set(stopwords.words('english')))
    
stops.update(['snr','oder','kommen','neuen','no','sachnummern',
                   'führt','aber','vorgesehen','bis','current','teile',
                    'gemäß','erfolgt','änderung','at','aktuellen',
                   'from','currently','as','parts','vergeben','sa',
                   'notwendig','zwei','müssen','lieferanten','release','bauteil',
                   'soll','verwendet','werk','hi','was','dieser',
                    'part','erforderlich','position','line','dies','kommt',
                   'serie','lieferant','varianten','mm','be','fahrzeuge',
                   'war','dadurch','bündel','unter','benötigt','00',
                   'vergabestand','mehr','fahrzeug','alle','001','bisher',
                   'variante','können','ab','änderungen','auch','bzw',
                    'aktuelle','china','bauteile','möglich','on','zeus',
                   'muss','neue','siehe','dokumentiert','dass','are',
                    'mopf','einem','da','not','einen','reifegrad',
                   'br','blank','wurden','haben','vorhanden','verbaut',
                    'gibt','über','diese','kein','vergabe','um',
                   'code','derzeit','einer','fzg','with','stand',
                   'zwischen','vom','aufgrund','als','bereich','ec',
                   'nur','sich','freigabe','noch','ohne','freigegeben',
                   'and','to','hat','nach','for','zb',
                    'wurde','beim','of','is','durch','zum',
                   'kann','keine','ein','am','aus','es',
                    'aktuell','dem','zur','von','eine','das',
                   'sind','wird','an','the','bei','auf',
                   'werden','zu','des','den','im','ist',
                   'nicht','mit','für','in','und','die',
                    'der','de','waren','übernahme','worden','dafür',
                   'hierzu','bekommt','jedoch','wegen','baureihen','baureihe',
                   'heute','stattfinden','geöffnetem','geöffneten','neben','geschehen',
                   'beigelegt','select','sollten','well','geänderte','aktueller',
                   'somit','bundel','hinzu','zwei','bereits','seit',
                   'bnd','lasst','beide','beiden','neu','neuer',
                   'alt','alter','alten','alte','liegt','dieseein',
                    'ec','freigabe','blankfreigegeben','change','neues','optimization',
                   'to','edition','optimierungen','code','entfall','standardbündel',
                   'standard','doku','mopf','freigeben','falsch','angepasst',
                   'berücksicht','berucksicht','teilweis','beid','entsprech',
                    'ausreich','teil','released','design','per','http'
                    'intra','corpintra','net','itp','html','yyyy'])
  


# In[715]:


import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from gensim import utils

from nltk.stem import SnowballStemmer

import spacy
nlp = spacy.load('de_core_news_sm')


# ### Replacement of Abbreviations

# In[716]:


df_sbreviation_list = pd.read_excel("abbreviation_ecr_des_20200511.xlsx")
df_sbreviation_list.columns = ['Abkuerzung', 'Bedeutung']
s_sbreviation_list = set(df_sbreviation_list['Abkuerzung'].map(lambda s: s.lower()))


# In[717]:


d_all_id_sbreviations = {}
for index,row in df_sbreviation_list.iterrows():
    if row['Abkuerzung'].lower() in s_sbreviation_list:
        
        d_all_id_sbreviations[row['Abkuerzung'].lower()] = row['Bedeutung'].lower().replace(u'\xa0', u'')


# In[718]:


def pre_sbv (s):
    s = s.lower()
    s = gsp.strip_punctuation(s)
    return s

# Hinweis: Wenn Abkürzungen Leerzeichen beinhalten (bsp. a / c) werden diese noch nicht korrigiert
def replace_with_sbreviations(d):
    ds = d.split()
    new_d = []
    for t in ds:
        if t.lower() in d_all_id_sbreviations:
            new_d.append(d_all_id_sbreviations[t.lower()])
        else:
            new_d.append(t)
    return " ".join(new_d)


# Hinweis: Wenn Abkürzungen Leerzeichen beinhalten (bsp. a / c) werden diese noch nicht korrigiert
def note_replaced_sbreviations(d):
    ds = d.split()
    replaced = []
    for t in ds:
        if t.lower() in d_all_id_sbreviations:
            replaced.append(t.lower())
    return set(replaced)


# In[719]:


from gensim import utils
import gensim.parsing.preprocessing as gsp


stemmer = SnowballStemmer('german')

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.strip_short
           #gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    
    s = ' '.join([token.lemma_ for token in nlp(s)])
    s = ' '.join([word for word in s.split() if word not in stops]) # delete stopwords from text
    s = ' '.join([stemmer.stem(word) for word in s.split()]) # stem german words
    return s


# In[720]:


input_df_final['new_IST_rep'] = input_df_final['new_IST'].apply(lambda x: pre_sbv(x))
input_df_final['IST_replaced_sb'] = input_df_final['new_IST_rep'].map(note_replaced_sbreviations)
input_df_final['new_IST_rep'] = input_df_final['new_IST_rep'].map(replace_with_sbreviations)

input_df_final['new_SOLL_rep'] = input_df_final['new_SOLL'].apply(lambda x: pre_sbv(x))
input_df_final['SOLL_replaced_sb'] = input_df_final['new_SOLL_rep'].map(note_replaced_sbreviations)
input_df_final['new_SOLL_rep'] = input_df_final['new_SOLL_rep'].map(replace_with_sbreviations)

input_df_final['Stichwort_rep'] = input_df_final['STICHWORT'].apply(lambda x: pre_sbv(x))
input_df_final['Stichwort_replaced_sb'] = input_df_final['Stichwort_rep'].map(note_replaced_sbreviations)
input_df_final['Stichwort_rep'] = input_df_final['Stichwort_rep'].map(replace_with_sbreviations)

input_df_final['Benennung_rep'] = input_df_final['BENENNUNG'].apply(lambda x: pre_sbv(x))
input_df_final['Benennung_replaced_sb'] = input_df_final['Benennung_rep'].map(note_replaced_sbreviations)
input_df_final['Benennung_rep'] = input_df_final['Benennung_rep'].map(replace_with_sbreviations)


# In[721]:


input_df_final['cleaned_IST'] = input_df_final['new_IST_rep'].apply(lambda x: clean_text(x))
input_df_final['cleaned_SOLL'] = input_df_final['new_SOLL_rep'].apply(lambda x: clean_text(x))
input_df_final['cleaned_Stichwort'] = input_df_final['Stichwort_rep'].apply(lambda x: clean_text(x))
input_df_final['cleaned_Benennung'] = input_df_final['Benennung_rep'].apply(lambda x: clean_text(x))


# In[722]:


input_df_final.columns


# In[723]:


lem_s = 'das abschirmblech hat im hinteren bereich eine kleine kollision mit dem tankspannband und eine engstelle zur abgasanlage fa autoneum herr hielscher'
lem = ' '.join([token.lemma_ for token in nlp(lem_s)])


# In[724]:


lem


# In[725]:


stop_s = 'der abschirmblech haben im hinter bereich einen kleine kollision mit der tankspannband und einen engstelle zur abgasanlage '
stop_result = ' '.join([word for word in stop_s.split() if word not in stops]) # delete stopwords from text
#s = ' '.join([stemmer.stem(word) for word in s.split()]) # stem german words
stop_result


# In[726]:


stem_s = 'abschirmblech kleine kollision tankspannband engstelle abgasanlage'
stem_result = ' '.join([stemmer.stem(word) for word in stem_s.split()]) # stem german words
stem_result


# In[ ]:





# In[727]:


input_df_final.loc[input_df_final['new_IST'].str.contains('AGA')]


# In[728]:


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

input_df_final.loc[input_df_final['BUENDEL_NUMMER']=='0045524-002']


# In[729]:


input_df_final.columns


# In[730]:


input_df_final_clean = input_df_final[['BUENDEL_NUMMER', 'CREATION_DATE', 'RUECKMELDE_DATUM',
       'STATUS', 'BENENNUNG', 'STICHWORT', 'VERURSACHER','new_IST', 'new_SOLL',
       'GEWUENSCHTER_EINSATZ', 'KOSTEN_RELEVANT', 'ZERTIFIZIERUNGS_RELEVANT',
       'PROZESS_STATUS', 'VERANTWORTLICHER_FK', 'GREMIUM_ENTSCHEIDUNG_MANUELL',
       'STATUS_GENEHMIGUNG', 'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART',
       'BEGR_RUECKSPRUNG', 'Bnd_MK', 'Bnd_MK_BENENNUNG', 'DAUER',
       'DAUER_KAT', 'TypZahl','MK','MSK', 'Bnd_KAT', 'KG', 'KGU',
       'cleaned_IST', 'cleaned_SOLL','cleaned_Stichwort', 'cleaned_Benennung',
        'ANZAHL_SACHNUMMER', 'ANZAHL_ZB_BENENNUNG']]


# ### Building the Machine Learning model & pipeline for Multilabel Classification
# 
# After all data exploration, let’s concentrate now on building the actual model. As it is a hierarchical multi-label classification, we need to convert our target label into a binarised vector with multiple bits set as 1.
# 
# ‘MultiLabelBinarizer’ of ‘scikit-learn’ can do that

# In[731]:


# delete duplicate strings in column ProduktArt 
for i, row in input_df_final_clean.iterrows():
    temp_typ = ';'.join(set(row['TypZahl'].split(sep=';')))
    input_df_final_clean.at[i,'TypZahl'] = temp_typ
    
# delete duplicate strings in column MK 
for i, row in input_df_final_clean.iterrows():
    temp_mk = ';'.join(set(row['MK'].split(sep=';')))
    input_df_final_clean.at[i,'MK'] = temp_mk



# delete duplicate strings in column KG
for i, row in input_df_final_clean.iterrows():
    temp_kg = ';'.join(set(row['KG'].split(sep=';')))
    input_df_final_clean.at[i,'KG'] = temp_kg


    
# delete duplicate strings in column KGU
for i, row in input_df_final_clean.iterrows():
    temp_kgu = ';'.join(set(row['KGU'].split(sep=';')))
    input_df_final_clean.at[i,'KGU'] = temp_kgu
    
# delete duplicate strings in column SUBK 
for i, row in input_df_final_clean.iterrows():
    temp_msk = ';'.join(set(row['MSK'].split(sep=';')))
    input_df_final_clean.at[i,'MSK'] = temp_msk
       


# In[732]:


input_df_final_clean.columns


# In[733]:


#input_df_final_clean['MK_KG'] = input_df_final_clean[['MK_BENENNUNG', 'KG']].apply(lambda x: ';'.join(x), axis=1)
input_df_final_clean['MK_KG'] = input_df_final_clean[['MK', 'KG']].apply(lambda x: ';'.join(x), axis=1)


# In[734]:


input_df_final_clean['MSK_KGU'] = input_df_final_clean[['MSK', 'KGU']].apply(lambda x: ';'.join(x), axis=1)


# In[735]:


input_df_final_clean['Target'] = input_df_final_clean[['MK_KG', 'MSK_KGU']].apply(lambda x: ';'.join(x), axis=1)


# ### Statistics of Data Source

# In[736]:


copy_df_kg = pd.DataFrame()
copy_df_kg['KG'] = input_df_final_clean['KG'].copy()
copy_df_kg['KG_count'] = copy_df_kg['KG'].apply(lambda x: len(str(x).split(";")))
tmp_buendel_kg_count = pd.DataFrame(copy_df_kg.groupby('KG_count').size())
tmp_buendel_kg_count


# In[737]:


copy_df_kg['KG_count'].mean()


# In[738]:


copy_df_mk = pd.DataFrame()
copy_df_mk['MK'] = input_df_final_clean['MK'].copy()
copy_df_mk['MK_count'] = copy_df_mk['MK'].apply(lambda x: len(str(x).split(";")))
tmp_buendel_mk_count = pd.DataFrame(copy_df_mk.groupby('MK_count').size())
tmp_buendel_mk_count


# In[739]:


copy_df_mk['MK_count'].mean()


# In[740]:


copy_df_msk = pd.DataFrame()
copy_df_msk['MSK'] = input_df_final_clean['MSK'].copy()
copy_df_msk['MSK_count'] = copy_df_msk['MSK'].apply(lambda x: len(str(x).split(";")))
tmp_buendel_msk_count = pd.DataFrame(copy_df_msk.groupby('MSK_count').size())
tmp_buendel_msk_count


# In[741]:


copy_df_msk['MSK_count'].mean()


# In[742]:


copy_df_mk_kg = pd.DataFrame()
copy_df_mk_kg['MK_KG'] = input_df_final_clean['MK_KG'].copy()
copy_df_mk_kg['MK_KG_count'] = copy_df_mk_kg['MK_KG'].apply(lambda x: len(str(x).split(";")))


# In[743]:


tmp_buendel_mk_kg_count = pd.DataFrame()
tmp_buendel_mk_kg_count['ECR_count']= copy_df_mk_kg.groupby('MK_KG_count').size()
tmp_buendel_mk_kg_count


# In[744]:


copy_df_mk_kg['MK_KG_count'].mean()


# In[745]:


copy_df_msk_kgu = pd.DataFrame()
copy_df_msk_kgu['MSK_KGU'] = input_df_final_clean['MSK_KGU'].copy()
copy_df_msk_kgu['MSK_KGU_count'] = copy_df_msk_kgu['MSK_KGU'].apply(lambda x: len(str(x).split(";")))


# In[746]:


copy_df_msk_kgu['MSK_KGU_count'].mean()


# In[747]:


tmp_buendel_msk_kgu_count = pd.DataFrame()
tmp_buendel_msk_kgu_count['ECR_count']= copy_df_msk_kgu.groupby('MSK_KGU_count').size()
tmp_buendel_msk_kgu_count


# In[748]:


input_df_final_clean['MSK_KGU']


# In[749]:


input_df_final_clean['MSK_KGU']


# In[750]:


# plot the distribution of KG among Bündles (when Bundle's istzustand is not NULL)
plt.figure(figsize=(12,7))

ax = tmp_buendel_mk_kg_count['ECR_count'].plot(kind='bar')

rects = ax.patches

for rect in rects:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 3
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    #label = "{:.1f}".format(y_value)
    label=y_value

    # Create annotation
    plt.annotate(                     # Use `label` as label
        label,
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }


plt.xlabel("Cardinality of upper-level categories",fontdict=font)
plt.ylabel("Number of ECRs",fontdict=font)

plt.show()


# In[ ]:





# In[757]:


copy_df_target = pd.DataFrame()
copy_df_target['Target'] = input_df_final_clean['Target'].copy()
copy_df_target['Target_count'] = copy_df_target['Target'].apply(lambda x: len(str(x).split(";")))

tmp_buendel_target_count = pd.DataFrame()
tmp_buendel_target_count['ECR_count'] = copy_df_target.groupby('Target_count').size()


# In[758]:


copy_df_target


# In[759]:


tmp_buendel_target_count


# In[760]:


copy_df_target['Target_count'].mean()


# In[761]:


# plot the distribution of KG among Bündles (when Bundle's istzustand is not NULL)
plt.figure(figsize=(12,7))

ax = tmp_buendel_msk_kgu_count['ECR_count'].plot(kind='bar')

rects = ax.patches

for rect in rects:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 3
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    #label = "{:.1f}".format(y_value)
    label=y_value

    # Create annotation
    plt.annotate(                     # Use `label` as label
        label,
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }


plt.xlabel("Cardinality of lower-level categories",fontdict=font)
plt.ylabel("Number of ECRs",fontdict=font)

plt.show()


# In[762]:


# plot the distribution of KG among Bündles (when Bundle's istzustand is not NULL)
plt.figure(figsize=(12,7))

ax = tmp_buendel_target_count['ECR_count'].plot(kind='bar')

rects = ax.patches

for rect in rects:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 3
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    #label = "{:.1f}".format(y_value)
    label=y_value

    # Create annotation
    plt.annotate(                     # Use `label` as label
        label,
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }


plt.xlabel("Cardinality of categories (all levels)",fontdict=font)
plt.ylabel("Number of ECRs",fontdict=font)

plt.show()


# In[763]:


df_buendel_linked.groupby(['VERANTWORTLICHER_FK'])['SACHNUMMER'].nunique().sort_values(ascending=False).head(10)


# In[764]:


input_df_final_clean.columns


# In[765]:


df_buendel_mk.groupby(['MK'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[766]:


df_count_mk = pd.DataFrame()
df_count_mk['MK_count'] = df_buendel_mk.groupby(['MK'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[767]:


df_count_mk


# In[768]:


# plot the distribution of KG among Bündles (when Bundle's istzustand is not NULL)
plt.figure(figsize=(14,7))

ax = df_count_mk['MK_count'].plot(kind='bar')

rects = ax.patches

for rect in rects:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 3
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        # Invert space to place label below
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    #label = "{:.1f}".format(y_value)
    label=y_value

    # Create annotation
    plt.annotate(                     # Use `label` as label
        label,
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va)                      # Vertically align label differently for
                                    # positive and negative values.
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }


plt.xlabel("Module Component",fontdict=font)
plt.ylabel("Number of ECRs",fontdict=font)

plt.show()


# In[769]:


df_buendel_mk['BUENDEL_NUMMER'].nunique()


# In[770]:


copy_df_target = pd.DataFrame()
copy_df_target['Target'] = input_df_final_clean['Target'].copy()
copy_df_target['Target_count'] = copy_df_target['Target'].apply(lambda x: len(str(x).split(" ")))

tmp_buendel_target_count = pd.DataFrame()
tmp_buendel_target_count['ECR_count'] = copy_df_target.groupby('Target_count').size()


# In[771]:


df_buendel_linked.groupby(['VERANTWORTLICHER_FK'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False).head(10)


# In[772]:


df_buendel_linked.groupby('VERANTWORTLICHER_FK')['SACHNUMMER'].nunique().sort_values(ascending=False).mean()


# In[773]:


df_buendel_linked.groupby(['BUENDEL_NUMMER'])['ZB_BENENNUNG'].nunique()


# In[774]:


df_buendel_linked.groupby(['BUENDEL_NUMMER'])['SACHNUMMER'].nunique().mean()


# In[ ]:





# In[775]:


input_df_final_clean.nunique()


# In[776]:


index_empty_zerti = input_df_final_clean.loc[input_df_final_clean['ZERTIFIZIERUNGS_RELEVANT']=='EMPTY'].index.tolist()
index_empty_zerti


# In[777]:


input_df_final_clean.drop(index_empty_zerti,inplace=True)


# In[778]:


input_df_final_clean.loc[input_df_final_clean['ZERTIFIZIERUNGS_RELEVANT']=='EMPTY']


# In[779]:


input_df_final_clean.columns


# In[780]:


input_df_final_clean.nunique()


# In[781]:


input_df_final_clean.groupby(['GREMIUM_ENTSCHEIDUNG_MANUELL'])['BUENDEL_NUMMER'].nunique()


# ### Build Feature Extractor for text and categorial data

# In[782]:


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
        


# In[846]:


class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
        fh_ec_verursacher = FeatureHasher(n_features=30, input_type='string')
        fh_ec_mk = FeatureHasher(n_features=103, input_type='string')
        fh_verantwortlicher = FeatureHasher(n_features=1100, input_type='string')
        #fh_zerti = FeatureHasher(n_features=2, input_type='string')
        #fh_cost = FeatureHasher(n_features=2, input_type='string')
        #fh_grem = FeatureHasher(n_features=40, input_type='string')

        ec_verursacher_vectorizer = fh_ec_verursacher.transform(X['VERURSACHER'])       
        ec_mk_vectorizer = fh_ec_mk.transform(X['Bnd_MK_BENENNUNG'])
        verantwort_vectorizer = fh_verantwortlicher.transform(X['VERANTWORTLICHER_FK'])
        #zerti_vectorizer = fh_zerti.transform(X['ZERTIFIZIERUNGS_RELEVANT'])
        #cost_vectorizer = fh_cost.transform(X['KOSTEN_RELEVANT'])
        #grem_vectorizer = fh_grem.transform(X['GREMIUM_ENTSCHEIDUNG_MANUELL'])


        return np.hstack(((np.asmatrix(ec_verursacher_vectorizer.toarray())), 
                         (np.asmatrix(ec_mk_vectorizer.toarray())),
                         (np.asmatrix(verantwort_vectorizer.toarray())),
                         #(np.asmatrix(zerti_vectorizer.toarray())),
                         #(np.asmatrix(cost_vectorizer.toarray())),
                         #(np.asmatrix(grem_vectorizer.toarray()))
                         ))


# In[784]:


# Create the tf-idf representation using the bag-of-words matrix
class TextBowTfidfTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self,field=None, min_df=None, stop_words=None):
        self.field = field
        self.min_df = min_df
        
    def fit(self, X, y=None):
        return self
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
        bow_transform = CountVectorizer(min_df=self.min_df, stop_words=stops)
        #bow_transform = CountVectorizer(min_df = self.min_df, stop_words=stops, ngram_range=(2,2))
        train_x_bow = bow_transform.fit_transform(train_x[str(self.field)])
        X_bow= bow_transform.transform(X[str(self.field)])
        
        tfidf_trfm = TfidfTransformer(norm=None)
        train_x_tfidf = tfidf_trfm.fit_transform(train_x_bow)
        X_tfidf = tfidf_trfm.transform(X_bow)
       
        return np.asmatrix(X_tfidf.toarray())


# In[785]:


#Categrical features to pass down the categorical pipeline 
categorical_features = []
categorical_features = ['VERURSACHER', 'Bnd_MK_BENENNUNG','VERANTWORTLICHER_FK','ZERTIFIZIERUNGS_RELEVANT',
                        'KOSTEN_RELEVANT','GREMIUM_ENTSCHEIDUNG_MANUELL']


#Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', SelectColumns(categorical_features) ),
                                           ( 'cat_transformer', CategoricalTransformer() ) ] )


# In[786]:


#Tf-idf: 
text_pipeline = FeatureUnion(transformer_list=[ ('benennung_trans', TextBowTfidfTransformer(field='cleaned_Benennung', min_df=3)),
                                                ('stichwort_trans', TextBowTfidfTransformer(field='cleaned_Stichwort', min_df=3)),
                                                ('ist_trans', TextBowTfidfTransformer(field='cleaned_IST', min_df=2))])


# In[787]:


# numeric features for predicting processing time of Bündel
numeric_features = []
numeric_features = ['ANZAHL_SACHNUMMER','ANZAHL_ZB_BENENNUNG']
numeric_pipeline = SelectColumns(numeric_features)


# In[788]:


preprocessing_pipeline = FeatureUnion( transformer_list = [ ( 'cat_pip', categorical_pipeline), 
                                                            ( 'text_pip', text_pipeline)] )


preprocessing_pipeline_it = FeatureUnion( transformer_list = [ ( 'cat_pip', categorical_pipeline), 
                                                            ( 'text_pip', text_pipeline),
                                                            ('num_pip',numeric_pipeline)] )


# #### (New)  preprocessing pipeline

# In[849]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
# Categorical with low-to-moderate cardinality
MAX_OH_CARDINALITY = 45

def select_oh_features(df):
    
    oh_features =        df        .select_dtypes(['object', 'category'])        .apply(lambda col: col.nunique())        .loc[lambda x: x <= MAX_OH_CARDINALITY]        .index        .tolist()
        
    return oh_features

#remove_cat_features = ['STATUS']
oh_features = select_oh_features(input_df_final_clean)

oh_features.remove('STATUS')
oh_features.remove('PROZESS_STATUS')
oh_features.remove('STATUS_GENEHMIGUNG')
oh_features.remove('KOSTENBEWERTUNGS_ART')
#oh_features.remove('Bnd_MK')
oh_features.remove('DAUER_KAT')
oh_features.remove('Bnd_KAT')
oh_features.remove('ZERTIFIZIERUNGS_RELEVANT')
oh_features.remove('KOSTEN_RELEVANT')
oh_features.remove('VERURSACHER')

#oh_features.remove('GREMIUM_ENTSCHEIDUNG_MANUELL')

#oh_features.remove('WERKZEUGAENDERUNG_ERFORDERLICH')
oh_features.append('VERANTWORTLICHER_FK')
oh_features.append('Bnd_MK_BENENNUNG')

#oh_pipeline = make_pipeline(SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore'))
oh_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

print(f'N oh_features: {len(oh_features)} \n')
print(', '.join(oh_features))


# In[790]:


import category_encoders as ce
#hc_features = ['VERANTWORTLICHER_FK', 'VERURSACHER', 'Bnd_MK_BENENNUNG', 'GREMIUM_ENTSCHEIDUNG_MANUELL']
hc_features = ['Bnd_MK_BENENNUNG']
hc_pipeline = make_pipeline(ce.TargetEncoder())
# hc_pipeline_verantwortlicher = make_pipeline(ce.HashingEncoder(n_components=1000))
# hc_pipeline_verursacher = make_pipeline(ce.HashingEncoder(n_components=30))
# hc_pipeline_mk = make_pipeline(ce.HashingEncoder(n_components=100))
# hc_pipeline_gre = make_pipeline(ce.HashingEncoder(n_components=40))


print(f'N hc_features: {len(hc_features)} \n')
print(', '.join(hc_features))


# In[850]:


from sklearn.compose import ColumnTransformer
column_transformer =    ColumnTransformer(transformers=                          [('txt_ist_pipeline', TfidfVectorizer(min_df=2,stop_words=stops), 'cleaned_IST'),                           ('txt_benennung_pipeline', TfidfVectorizer(min_df=3,stop_words=stops), 'cleaned_Benennung'),                           ('txt_stichwort_pipeline', TfidfVectorizer (min_df=3, stop_words=stops), 'cleaned_Stichwort'),                            ('oh_pipeline', OneHotEncoder(handle_unknown='ignore'), oh_features)],                            #('hc_pipeline', ce.TargetEncoder(), hc_features)],\
                            n_jobs=-1,remainder='drop')


# In[ ]:





# In[792]:


input_df_final_clean['DAUER_KAT'].describe()


# In[793]:


tar_labels = []
tar_labels = [
        row['MK_KG'].split(";")
        for index,row in input_df_final_clean.iterrows()
        ]


# In[794]:


tar_labels


# In[795]:


df_mk.loc[df_mk['MK_BENENNUNG'].str.contains('Motorlager/Getriebelager')]


# In[796]:


from sklearn.preprocessing import MultiLabelBinarizer

y = []    
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(tar_labels)


# In[797]:


y.shape


# In[798]:


mlb.classes_


# In[799]:


df_x = pd.DataFrame(input_df_final_clean.copy())


# In[800]:


# df_x['Bnd_MK_BENENNUNG'] = df_x['Bnd_MK_BENENNUNG'].astype('str')
# df_x['VERURSACHER'] = df_x['VERURSACHER'].astype('str')
# df_x['ZERTIFIZIERUNGS_RELEVANT'] = df_x['ZERTIFIZIERUNGS_RELEVANT'].astype('str')
# df_x['GREMIUM_ENTSCHEIDUNG_MANUELL'] = df_x['GREMIUM_ENTSCHEIDUNG_MANUELL'].astype('str')

df_x['cleaned_IST'] = df_x['cleaned_IST'].astype('str')
df_x['cleaned_Benennung'] = df_x['cleaned_Benennung'].astype('str')
df_x['cleaned_Stichwort'] = df_x['cleaned_Stichwort'].astype('str')

df_x['VERURSACHER'] = df_x['VERURSACHER'].fillna('None')
df_x['VERURSACHER'] = df_x['VERURSACHER'].astype('category')

df_x['KOSTEN_RELEVANT'] = df_x['KOSTEN_RELEVANT'].fillna('None')
df_x['KOSTEN_RELEVANT'] = df_x['KOSTEN_RELEVANT'].astype('category')

df_x['ZERTIFIZIERUNGS_RELEVANT'] = df_x['ZERTIFIZIERUNGS_RELEVANT'].fillna('None')
df_x['ZERTIFIZIERUNGS_RELEVANT'] = df_x['ZERTIFIZIERUNGS_RELEVANT'].astype('category')
#df_x['GREMIUM_ENTSCHEIDUNG_MANUELL'] = df_x['GREMIUM_ENTSCHEIDUNG_MANUELL'].astype('category')
df_x['Bnd_MK_BENENNUNG'] = df_x['Bnd_MK_BENENNUNG'].fillna('None')
df_x['Bnd_MK_BENENNUNG'] = df_x['Bnd_MK_BENENNUNG'].astype('category')
# df_x['EMISSIONSKENNZEICHEN'] = df_x['EMISSIONSKENNZEICHEN'].astype('category')
# df_x['BESCHAFFUNGSART'] = df_x['BESCHAFFUNGSART'].astype('category')
# df_x['SICHERHEITSRELEVANT'] = df_x['SICHERHEITSRELEVANT'].astype('category')
# df_x['TEILEART'] = df_x['TEILEART'].astype('category')
# df_x['WERKZEUGAENDERUNG_ERFORDERLICH'] = df_x['WERKZEUGAENDERUNG_ERFORDERLICH'].astype('category')
# df_x['EE_STEUERGERAET_BETROFFEN'] = df_x['EE_STEUERGERAET_BETROFFEN'].astype('category')
# df_x['BETRIEBSANLEITUNG_BETROFFEN'] = df_x['BETRIEBSANLEITUNG_BETROFFEN'].astype('category')
# df_x['LEITUNGSSATZ_BETROFFEN'] = df_x['LEITUNGSSATZ_BETROFFEN'].astype('category')
# df_x['LASTENHEFTAENDERUNG_ERFORDERLICH'] = df_x['LASTENHEFTAENDERUNG_ERFORDERLICH'].astype('category')
# df_x['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'] = df_x['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'].astype('category')
# df_x['DAUER_KAT_num'] = df_x['DAUER_KAT'].astype('category').cat.codes
df_x['MK_KG_num'] = df_x['MK_KG'].astype('category').cat.codes


# In[801]:


from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)

for train_index, test_index in msss.split(df_x, y):
    print("TRAIN:", train_index, "TEST:", test_index)


# In[223]:


np.savetxt("train_index_c_20201224.csv", train_index, delimiter=",")
np.savetxt("test_index_c_20201224.csv", test_index, delimiter=",")


# In[224]:


# train_index = np.loadtxt("train_index_ab_20200915.csv", dtype='i', delimiter=',')
# test_index =  np.loadtxt("test_index_ab_20200915.csv", dtype='i', delimiter=',')


# In[802]:


train_x, test_x = df_x.iloc[train_index], df_x.iloc[test_index]
train_y, test_y = y[train_index], y[test_index]


# In[803]:


test_y.shape


# In[840]:


train_y


# In[230]:


# cat_train = categorical_pipeline.fit_transform(train_x)
# cat_train


# In[844]:


df_x.dtypes


# ### Multilabel Classification

# In[804]:


from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from tqdm import tqdm_notebook as tqdm


# In[805]:


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


# ### Label Community Detection

# In[806]:


from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.cluster import NetworkXLabelGraphClusterer

graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)


# In[807]:


graph_builder.transform(train_y)


# In[808]:


# display as graph
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)

nlabel = train_y.shape[1]
label_names = mlb.classes_.copy()
edge_map = graph_builder.transform(train_y)
print("{} labels, {} edges".format(len(label_names), len(edge_map)))


# In[809]:


print(edge_map)


# In[810]:


from skmultilearn.cluster import IGraphLabelGraphClusterer
import igraph as ig

# setup the clusterer to use, we selected the fast greedy modularity-maximization approach
clusterer_ig = IGraphLabelGraphClusterer(graph_builder=graph_builder, method='fastgreedy')
partition_ig = clusterer_ig.fit_predict(train_x, train_y)


# In[811]:


mlb.classes_[71]


# In[812]:


partition_ig


# In[813]:


len(partition_ig)


# In[814]:


# we define a helper function for visualization purposes
def to_membership_vector(partition):
    return {
        member :  partition_id
        for partition_id, members in enumerate(partition)
        for member in members
    }


# In[815]:


label_names


# In[816]:


mlb.classes_


# In[817]:


df_buendel_linked[['MK','MK_BENENNUNG']]


# In[841]:


df_buendel_linked.columns


# In[819]:


for i,element in enumerate(label_names):
    label_names[i] = label_names[i].replace('KG_01','KG_Motorgehäuse') 
    label_names[i] = label_names[i].replace('KG_09','KG_Kraftstoffförder- / Ladepumpe') 
    label_names[i] = label_names[i].replace('KG_24','KG_Motor- / Aggregateaufhängung im Fahrgestell')
    label_names[i] = label_names[i].replace('KG_27','KG_Automat. / Hydr. Getriebe')
    
    label_names[i] = label_names[i].replace('KG_32','KG_Federn und Federaufhängung')
    label_names[i] = label_names[i].replace('KG_33','KG_Vorderachse')
    
    label_names[i] = label_names[i].replace('KG_34','KG_Altern. Antriebe/ Energieumw.')
    label_names[i] = label_names[i].replace('KG_35','KG_Hinterachse')
    label_names[i] = label_names[i].replace('KG_40','KG_Räder und Bereifung')
    label_names[i] = label_names[i].replace('KG_41','KG_Gelenkwelle / Achsantrieb')
    label_names[i] = label_names[i].replace('KG_42','KG_Bremse')
    label_names[i] = label_names[i].replace('KG_46','KG_Lenkung')
    label_names[i] = label_names[i].replace('KG_47','KG_Kraftstoffanlage')
    label_names[i] = label_names[i].replace('KG_49','KG_Abgas- / Abluftanlage')
    label_names[i] = label_names[i].replace('KG_50','KG_Kühler / Wärmetauscher')
    label_names[i] = label_names[i].replace('KG_52','KG_Fahrgestellanbauteile')
    label_names[i] = label_names[i].replace('KG_54','KG_Elektr. Ausrüstung Fahrgestell')
    label_names[i] = label_names[i].replace('KG_58','KG_Werkzeug und Zubehör')
    label_names[i] = label_names[i].replace('KG_60','kG_Aufbau')
    label_names[i] = label_names[i].replace('KG_61','KG_Unterbau')
    label_names[i] = label_names[i].replace('KG_62','KG_Vorderwand')
    label_names[i] = label_names[i].replace('KG_63','KG_Seitenwände')
    label_names[i] = label_names[i].replace('KG_64','KG_Rückwand, Heck')
    label_names[i] = label_names[i].replace('KG_65','KG_Dach')
    label_names[i] = label_names[i].replace('KG_67','KG_Fensteranlage')
    label_names[i] = label_names[i].replace('KG_68','KG_Verkleidung_A')
    label_names[i] = label_names[i].replace('KG_69','KG_Verkleidung_B')
    label_names[i] = label_names[i].replace('KG_72','KG_Seitentür vorn')
    label_names[i] = label_names[i].replace('KG_73','KG_Seitentür hinten, Mitteltür')
    label_names[i] = label_names[i].replace('KG_74','KG_Rückwandtür')
    label_names[i] = label_names[i].replace('KG_75','KG_Aussenklappen')
    label_names[i] = label_names[i].replace('KG_77','KG_Verdeck')
  
    label_names[i] = label_names[i].replace('KG_78','KG_Dachsysteme')
    label_names[i] = label_names[i].replace('KG_79','KG_Aufsätze, Spriegelk., Klappdach')
    
    label_names[i] = label_names[i].replace('KG_81','KG_Ausstattung, Inneneinrichtung')
    label_names[i] = label_names[i].replace('KG_82','KG_Elektrische Anlage')
    label_names[i] = label_names[i].replace('KG_83','KG_Lüftung, Heizung, Klimatisierung')
    label_names[i] = label_names[i].replace('KG_86','KG_Sicherheitssys. Notger. San.Ausr.')
    label_names[i] = label_names[i].replace('KG_88','KG_Kotflügel, Motorhaube, Stoßfänger usw')
    label_names[i] = label_names[i].replace('KG_89','KG_Sondereinb., Fahrzg.-Zub.')
    
    label_names[i] = label_names[i].replace('KG_90','KG_Elektrik- / Elektronik-Komponenten (Pkw)')
    label_names[i] = label_names[i].replace('KG_91','KG_Vordersitze, Fahrersitz')
    label_names[i] = label_names[i].replace('KG_92','KG_Rücksitz, Sitzbank')
    label_names[i] = label_names[i].replace('KG_97','KG_Sitzzubehör, Liege')
    label_names[i] = label_names[i].replace('KG_98','KG_Allgemein / KG übergreifende Teile')
    label_names[i] = label_names[i].replace('KG_99','KG_Normähnliche Teile')


# In[820]:


colors = ['lightpink', 'white', 'steelblue','green','yellow','cyan','magenta','gold','skyblue','violet','darkkhaki','darkcyan','hotpink','azure']
membership_vector = to_membership_vector(partition_ig)
visual_style = {
    "vertex_size" : 20,
    "vertex_label": [x for x in label_names],
    "vertex_label_size": 8,
    "vertex_label_dist": 1,
    "edge_width" : [80*x/train_y.shape[0] for x in clusterer_ig.graph_.es['weight']],
    "vertex_color": [colors[membership_vector[i]] for i in range(train_y.shape[1])],
    "bbox": (1200,1200),
    "margin": 80,
    #"layout": clusterer_ig.graph_.layout_drl()
    "layout": 'large'

}

ig.plot(clusterer_ig.graph_, **visual_style)


# In[821]:


ig.plot(clusterer_ig.graph_, "community_ab.pdf", **visual_style)


# In[822]:


ig.plot(clusterer_ig.graph_, "community_ab.png", **visual_style)


# In[823]:


membership_vector


# In[824]:


membership_vector[6]


# In[825]:


dict_community_label = dict(enumerate(label_names))


# In[826]:


dict_community_label


# In[827]:


dict_membership_vector = dict((dict_community_label[key], value) for (key, value) in membership_vector.items())
dict_membership_vector


# In[828]:


# test_txt_pre = text_pipeline.fit_transform(test_x)
# test_txt_pre.shape


# In[829]:


txt_pre = text_pipeline.fit_transform(train_x)
txt_pre.shape


# In[ ]:





# In[847]:


text_cat_pre = categorical_pipeline.fit_transform(train_x)
text_cat_pre.shape


# In[ ]:


# num_pre = numeric_pipeline.fit_transform(train_x)
# num_pre.shape


# In[ ]:


#df_pre = column_transformer.fit_transform(df_x, df_x['MK_KG_num'])


# In[ ]:





# In[851]:


df_pre = column_transformer.fit_transform(df_x)

# df_pre = column_transformer.fit_transform(df_x_it)
df_pre.shape


# In[852]:


train_pre = column_transformer.transform(train_x)
train_pre.shape


# In[853]:


test_pre = column_transformer.transform(test_x)
test_pre.shape


# In[260]:


#df_x.nunique()


# In[854]:


test_y.shape


# In[856]:


# dimension of categorical attributes: 
columns_txt = np.array(range(0,5498))
columns_cat = np.array(range(5498,7578))


# In[ ]:


train_pre.toarray()[0][11700:11710]


# In[ ]:


#del lsp, lsp_model


# In[ ]:


#del pred_lsp


# In[ ]:


#lsp.classifiers_[4].classifier.clfs_[0][1].estimators_[0]


# In[ ]:





# In[857]:


from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain
import time
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.multioutput import ClassifierChain

# #base_classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=16, verbose=True)
lg = LogisticRegression(max_iter=4000)
#lsvc = LinearSVC()
base_rf = RandomForestClassifier(bootstrap=False, criterion='gini', 
                                 random_state=42, n_estimators=100, n_jobs=-1, verbose=True)

#gb = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=True)
#clf_svm = svm.SVC(random_state=42, verbose=True)
#mnb = MultinomialNB()
#knn = KNeighborsClassifier(n_jobs=16)

pip_cat = make_pipeline(ColumnSelector(cols=(columns_cat)),base_rf)
pip_txt = make_pipeline(ColumnSelector(cols=(columns_txt)),base_rf)

sclf_cv = StackingCVClassifier(classifiers=[pip_cat, pip_txt], 
                          meta_classifier=lg, use_probas=True, random_state=42, cv=3)


# problem transformation from multi-label to single-label multi-class
transformation_classifier = LabelPowerset(sclf_cv)
#transformation_classifier = BinaryRelevance(sclf_cv)
#transformation_classifier = LabelPowerset(base_rf)
#transformation_classifier = ClassifierChain(sclf_cv)

# ensemble
lsp = LabelSpacePartitioningClassifier(transformation_classifier, clusterer_ig)
#lsp = LabelPowerset(base_rf)
#lsp = ClassifierChain(base_rf)

#lsp_model = make_pipeline(preprocessing_pipeline, lsp)
#lsp_model = make_pipeline(categorical_pipeline, lsp)

start=time.time()
#lsp_model.fit(train_x,train_y)
lsp.fit(train_pre,train_y)
print('training time taken: ',round(time.time()-start,0),'seconds')
#print (lsp_model.best_params_, lsp_model.best_score_)


# In[264]:


import joblib
joblib.dump(lsp, 'ab_lsp_20210119.pkl')


# In[837]:


train_x.to_csv("train_x_ab_20210119.csv",encoding='utf-8-sig')


# In[839]:





# In[838]:


df_buendel_linked.to_csv("df_buendel_linked_ab_20210119.csv",encoding='utf-8-sig')


# In[843]:


input_df_final_clean.to_csv("input_df_final_clean_ab_20210119.csv",encoding='utf-8-sig')


# In[661]:


# Gridsearch
import time
from skmultilearn.problem_transform import ClassifierChain
base_rf = RandomForestClassifier(bootstrap=False, criterion='gini', 
                                 random_state=42, n_estimators=100, n_jobs=-1, verbose=True)

#lsp_cv = LabelPowerset(base_rf)

#lsp_model_cv = make_pipeline(preprocessing_pipeline, lsp_cv)

parameters = [
    {
        'classifier': [base_rf]
    }
]

#score = 'f1_micro'
score = 'precision_micro'
#score = 'recall_micro'
#score = 'accuracy'
#score = make_scorer(metrics.hamming_loss, greater_is_better=False)
clf = GridSearchCV(BinaryRelevance(), parameters, scoring= score,cv=3)
clf.fit(train_pre, train_y)

start=time.time()
#lsp_model_cv.fit(train_x,train_y)
#lsp.fit(sparse_train_x_pre,train_y)
print('training time taken: ',round(time.time()-start,0),'seconds')
#print (lsp_model.best_params_, lsp_model.best_score_)


# In[317]:


sorted(metrics.SCORERS.keys())


# In[662]:


clf.cv_results_


# In[435]:


clf.classes_


# In[319]:


pred_lsp_cv = clf.predict(test_pre)


# In[431]:


pred_lsp_cv_proba = clf.predict_proba(test_pre)


# In[434]:


pred_lsp_cv_proba.shape


# In[858]:


pred_lsp= lsp.predict(test_pre)


# In[320]:


# Performance of original model
print("Accuracy_CV = ",round(accuracy_score(test_y,pred_lsp_cv.toarray()),ndigits=3))
print("Accuracy_perLabel_CV = ",round(hamming_score(test_y,pred_lsp_cv.toarray()),ndigits=3))
print("Hamming_Loss_Stacking_CV = ",round(metrics.hamming_loss(test_y,pred_lsp_cv),ndigits=3))
print('\n')

print("Micro_Precision_CV= ",round(precision_score(test_y, pred_lsp_cv, average='micro'),ndigits=3))
print("Precision_per class_CV = ",precision_score(test_y, pred_lsp_cv, average=None))
print('\n')

print("Micro_Recall_CV = ",round(recall_score(test_y, pred_lsp_cv, average='micro'),ndigits=3))
print("Weighted_Recall_CV = ",round(recall_score(test_y, pred_lsp_cv, average='weighted'),ndigits=3))
print('\n')

# Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. 
n_classes = train_y.shape[1] -1
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(test_y[:, i],
                                                        pred_lsp_cv.toarray()[:, i])
    average_precision[i] = average_precision_score(test_y[:, i], pred_lsp_cv.toarray()[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(test_y.ravel(),
    pred_lsp_cv.toarray().ravel())
average_precision["micro"] = average_precision_score(test_y, pred_lsp_cv.toarray(),
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

print("micro_F1 Score = ",round(f1_score(test_y, pred_lsp_cv, average='micro'),ndigits=3))
print("F1_Score_per class = ",f1_score(test_y, pred_lsp_cv, average=None))


# In[859]:


# Performance of original model
print("Accuracy_LSP_LP = ",round(accuracy_score(test_y,pred_lsp.toarray()),ndigits=3))
print("Accuracy_perLabel = ",round(hamming_score(test_y,pred_lsp.toarray()),ndigits=3))
print("Hamming_Loss_Stacking = ",round(metrics.hamming_loss(test_y,pred_lsp),ndigits=3))
print('\n')

print("Micro_Precision = ",round(precision_score(test_y, pred_lsp, average='micro'),ndigits=3))
print("Precision_per class = ",precision_score(test_y, pred_lsp, average=None))
print('\n')

print("Micro_Recall = ",round(recall_score(test_y, pred_lsp, average='micro'),ndigits=3))
print("Weighted_Recall = ",round(recall_score(test_y, pred_lsp, average='weighted'),ndigits=3))
print('\n')

# Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. 
n_classes = train_y.shape[1] -1
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(test_y[:, i],
                                                        pred_lsp.toarray()[:, i])
    average_precision[i] = average_precision_score(test_y[:, i], pred_lsp.toarray()[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(test_y.ravel(),
    pred_lsp.toarray().ravel())
average_precision["micro"] = average_precision_score(test_y, pred_lsp.toarray(),
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

print("micro_F1 Score = ",round(f1_score(test_y, pred_lsp, average='micro'),ndigits=3))
print("F1_Score_per class = ",f1_score(test_y, pred_lsp, average=None))


# In[848]:


mlb.classes_


# In[328]:


mlb.classes_[4]


# In[277]:


y.shape


# In[278]:


import warnings
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


# In[279]:


feature_names = np.array(get_feature_names(column_transformer))


# In[280]:


feature_names.shape


# In[281]:


feature_names[10708:11710]


# In[634]:


train_x.iloc[0:5]['VERURSACHER']


# ### Visualizting Random Forest

# In[344]:


clf.best_estimator_.classifiers_[4].classes_


# In[427]:


lsp.classifiers_[4].classifier.clfs_[0]


# In[326]:


lsp.classifier.classifier.classifiers[0]['randomforestclassifier']


# In[419]:


import os
import pydot
from sklearn.tree import export_graphviz

#class_names = ['NO_Außenbeleuchtung','Außenbeleuchtung']
class_names = str(lsp.classifiers_[4].classifier.clfs_[0][1].estimators_[0].classes_)

export_graphviz(lsp.classifiers_[4].classifier.clfs_[0][1].estimators_[0], out_file = 'tree.dot',
               feature_names=feature_names[10621:11711],
               class_names=class_names,
               rounded=True, proportion=False,
               precision=2, filled=True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_amg.png', '-Gdpi=600'])

# Display in jupyter notebook
# from IPython.display import Image
# Image(filename = 'tree.png')


# In[420]:


feature_names[100:140]


# In[426]:


train_x.loc[train_x['Bnd_MK_BENENNUNG'].str.contains('Cockpit')].head(6)[['new_IST']]


# ### Explanation

# In[438]:


test_sample = train_x.loc[train_x['new_IST'].str.contains('Das Abschirmblech hat im hinteren')]
test_sample_pre = column_transformer.transform(test_sample)


# In[436]:


from collections import OrderedDict
#from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

class_names = str(clf.best_estimator_.classifiers_[4].classes_)
#explainer = LimeTextExplainer(class_names=class_names)
explainer = LimeTabularExplainer(train_pre,feature_names=feature_names, class_names=class_names, discretize_continuous=True)
#explanation = explainer.explain_instance(train_it_pre, clf.predict_proba, num_features=6)

# explanation.as_pyplot_figure();
# plt.show()


# In[443]:


exp = explainer.explain_instance(test_sample_pre, clf.best_estimator_.classifiers_[4].predict_proba, num_features=5)


# In[444]:


exp.as_list()


# ### Calculating cross-validation results in each community

# In[307]:


partition_ig[0]


# In[335]:


# calculate result of CV in Community [4]

list_target_com_11 = mlb.classes_[partition_ig[11]].tolist()

train_x_com_11 = train_x[train_x['MK_KG'].str.contains('|'.join(list_target_com_11))]

#train_x_com_11_pre = preprocessing_pipeline.fit_transform(train_x_com_11)
train_x_com_11_pre = column_transformer.transform(train_x_com_11)

train_y_com_11 = train_x_com_11['MK_KG']


for clf_percommunity, label in zip([pip_cat, pip_txt, sclf_cv], 
                      ['CAT', 
                       'TXT', 
                       'StackingCVClassifier']):

    scores = model_selection.cross_val_score(clf_percommunity, train_x_com_11_pre, train_y_com_11, 
                                              cv=3, scoring='accuracy')
    print("F1 Score: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# In[357]:


clf_percommunity.classifiers[0].steps[1].index(1)


# In[358]:


sclf_cv


# In[857]:


train_x.loc[train_x['MK_KG'].str.contains('Zentrale Steuergeräte')]['BUENDEL_NUMMER'].nunique()


# In[858]:


test_x.loc[test_x['MK_KG'].str.contains('Zentrale Steuergeräte')]['BUENDEL_NUMMER'].nunique()


# In[701]:


mlb.classes_


# In[ ]:




