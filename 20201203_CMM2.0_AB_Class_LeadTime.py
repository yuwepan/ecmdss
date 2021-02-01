#!/usr/bin/env python
# coding: utf-8

# ## EC Processing Time Analysis -  A/B Class
# - Data Preprocessing
# - Focus on each product class
# - Analyzing Rücksprung Kommentar - frequent words (unigram/bigram/trigram)
# - Analyzing influencing factors of processing time of Bündels
# - Predicting processing time of new Bündel

# In[657]:


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

# In[658]:


# load EC releted entities
# EC_Entity, EC_Kategorie, Kategorie, Modulekomponente

home_dir_tables = "/home/yuwepan/Promotion/Data/cmm2.0/20201116/"
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


# In[659]:


df_ec.head()


# In[660]:


# load buendel related entities
# Buendel Entity, Planeinsatztermin, Paket, BR_AA, BuendelZustand, 
df_buendel_entity = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD151_BuendelEntity.parquet', engine='pyarrow')
df_buendel_entity = df_buendel_entity[['ID','CREATION_DATE','BUENDEL_NUMMER','STATUS','BENENNUNG',
                                       'GEWUENSCHTER_EINSATZ','KOSTEN_RELEVANT', 'ZERTIFIZIERUNGS_RELEVANT',
                                       'PROZESS_STATUS','EC_FK','VERANTWORTLICHER_FK','GREMIUM_ENTSCHEIDUNG_MANUELL',
                                       'STATUS_GENEHMIGUNG','RUECKSPRUNG_BEANTRAGT','RUECKSPRUNG_KOMMENTAR',
                                      'KOSTENBEWERTUNGS_ART','BEGR_RUECKSPRUNG','RUECKMELDE_DATUM','MODIFICATION_DATE',
                                      'MODULKOMPONENTE','BUENDEL_ZUSTAND_FK']]

df_buendel_entity.columns = ['Bnd_ID','Bnd_CREATION_DATE','BUENDEL_NUMMER','Bnd_STATUS','Bnd_BENENNUNG',
                                       'Bnd_GEWUENSCHTER_EINSATZ','Bnd_KOSTEN_RELEVANT', 'Bnd_ZERTIFIZIERUNGS_RELEVANT',
                                       'Bnd_PROZESS_STATUS','EC_FK','Bnd_VERANTWORTLICHER_FK','Bnd_GREMIUM_ENTSCHEIDUNG_MANUELL',
                                       'Bnd_STATUS_GENEHMIGUNG','Bnd_RUECKSPRUNG_BEANTRAGT','Bnd_RUECKSPRUNG_KOMMENTAR',
                                       'Bnd_KOSTENBEWERTUNGS_ART','Bnd_BEGR_RUECKSPRUNG','Bnd_RUECKMELDE_DATUM', 
                                       'Bnd_MODIFICATION_DATE','Bnd_MODULKOMPONENTE','BUENDEL_ZUSTAND_FK']

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
df_amount_paket = df_pet_paket_braa.groupby(['BUENDEL_FK'])['PAKET_FK'].nunique().reset_index()
df_amount_paket.columns = ['BUENDEL_FK','ANZAHL_PAKET']
#df_pet_paket_braa['ANZAHL_PAKET'] = df_amount_paket['ANZAHL_PAKET']
df_pet_paket_braa = df_pet_paket_braa.merge(df_amount_paket,left_on='BUENDEL_FK', right_on='BUENDEL_FK')

new_df_bnd_entity = df_pet_paket_braa.merge(df_buendel_entity,left_on='BUENDEL_FK', right_on='Bnd_ID',
                                    how='left')
new_df_bnd_entity = new_df_bnd_entity.merge(df_buendel_zustand,left_on='BUENDEL_ZUSTAND_FK',right_on='BUENDEL_ZUSTAND_ID')
new_df_bnd_entity = new_df_bnd_entity.merge(df_ec, left_on='EC_FK', right_on='EC_ID')

new_df_bnd_entity = new_df_bnd_entity.merge(df_mk,left_on='Bnd_MODULKOMPONENTE',right_on='MK')


# In[661]:


df_braa


# In[662]:


new_df_bnd_entity.columns


# In[663]:


new_df_bnd_entity['ANZAHL_PAKET'].describe()


# In[664]:


new_df_bnd_drop = new_df_bnd_entity [['EC_NUMMER', 'EC_STATUS', 'EC_STICHWORT',
       'EC_IST_ZUSTAND', 'EC_SOLL_ZUSTAND', 'EC_VERURSACHER',
       'EC_VERANTWORTLICHER_FK','EC_KATEGORIE',
       'KAT_BENENNUNG', 'Bnd_ID', 'Bnd_CREATION_DATE', 'BUENDEL_NUMMER', 'Bnd_STATUS',
       'Bnd_BENENNUNG', 'Bnd_GEWUENSCHTER_EINSATZ', 'Bnd_KOSTEN_RELEVANT',
       'Bnd_ZERTIFIZIERUNGS_RELEVANT', 'Bnd_PROZESS_STATUS', 'EC_FK_x',
       'Bnd_VERANTWORTLICHER_FK', 'Bnd_GREMIUM_ENTSCHEIDUNG_MANUELL',
       'Bnd_STATUS_GENEHMIGUNG', 'Bnd_RUECKSPRUNG_BEANTRAGT',
       'Bnd_RUECKSPRUNG_KOMMENTAR', 'Bnd_KOSTENBEWERTUNGS_ART',
       'Bnd_BEGR_RUECKSPRUNG', 'Bnd_RUECKMELDE_DATUM', 'Bnd_MODIFICATION_DATE','Bnd_MODULKOMPONENTE','MK_BENENNUNG_y',
       'Bnd_IST_ZUSTAND','Bnd_SOLL_ZUSTAND','BR','AA','ANZAHL_PAKET']].drop_duplicates(subset=None, keep='first')


# In[665]:


new_df_bnd_drop['BUENDEL_NUMMER'].nunique()


# In[666]:


new_df_bnd_drop['Bnd_KOSTEN_RELEVANT'] = new_df_bnd_drop['Bnd_KOSTEN_RELEVANT'].fillna('NaN')
new_df_bnd_drop['Bnd_ZERTIFIZIERUNGS_RELEVANT'] = new_df_bnd_drop['Bnd_ZERTIFIZIERUNGS_RELEVANT'].fillna('NaN')

new_df_bnd_drop['EC_STICHWORT'] = new_df_bnd_drop['EC_STICHWORT'].fillna('NaN')
new_df_bnd_drop['EC_IST_ZUSTAND'] = new_df_bnd_drop['EC_IST_ZUSTAND'].fillna('NaN')
new_df_bnd_drop['EC_SOLL_ZUSTAND'] = new_df_bnd_drop['EC_SOLL_ZUSTAND'].fillna('NaN')

new_df_bnd_drop['Bnd_BENENNUNG'] = new_df_bnd_drop['Bnd_BENENNUNG'].fillna('NaN')
new_df_bnd_drop['Bnd_RUECKSPRUNG_KOMMENTAR'] = new_df_bnd_drop['Bnd_RUECKSPRUNG_KOMMENTAR'].fillna('NaN')
new_df_bnd_drop['Bnd_IST_ZUSTAND'] = new_df_bnd_drop['Bnd_IST_ZUSTAND'].fillna('NaN')
new_df_bnd_drop['Bnd_SOLL_ZUSTAND'] = new_df_bnd_drop['Bnd_SOLL_ZUSTAND'].fillna('NaN')


# In[667]:


new_df_bnd_drop.loc[new_df_bnd_drop['Bnd_IST_ZUSTAND'].str.contains('siehe EC')]['BUENDEL_NUMMER'].nunique()


# In[668]:


new_df_bnd_drop.loc[new_df_bnd_drop['EC_IST_ZUSTAND'].str.contains('siehe die Beschreibung des Bündels')]['BUENDEL_NUMMER'].nunique()


# In[669]:


# load produkt, convert BR to product class
df_produkt = pd.read_csv('/home/yuwepan/Promotion/Data/Produkt.csv',sep="|",header=0)


# In[670]:


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
        


# In[671]:


df_produkt['new_ProduktKlasse'] = df_produkt['ProduktKlasse'].apply(lambda x: convert_pk(x))


# In[672]:


df_produkt


# In[673]:


df_produkt.nunique()


# In[674]:


df_produkt = df_produkt[['BR','new_ProduktKlasse']].drop_duplicates(subset=None, keep='first')
df_produkt


# In[675]:


# add column 'new_ProduktKlase' to merged buendel entity
new_df_bnd_drop_produkt = new_df_bnd_drop.merge(df_produkt,left_on='BR',right_on='BR')


# In[676]:


new_df_bnd_drop_produkt['new_ProduktKlasse'].nunique()


# In[677]:


new_df_bnd_drop_produkt.groupby(['BR'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[678]:


new_df_bnd_drop_produkt.groupby(['new_ProduktKlasse'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[679]:


new_df_bnd_drop_produkt.columns


# #### Add affected werk to Bündels

# In[680]:


df_betroffeneWerk_entity = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD186_BetroffeneWerke_Buendel.parquet', engine='pyarrow')
df_betroffenWerke_admin = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD187_BetroffenWerke_Admin.parquet', engine='pyarrow')
df_adminwerk_entity = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRCWLM203_AdminWerkEntity.parquet', engine='pyarrow')

df_werk = df_betroffeneWerk_entity.merge(df_betroffenWerke_admin, left_on='ID',right_on='BETROFFENE_WERK_FK')
df_bnd_werk = df_werk.merge(df_adminwerk_entity, left_on='ADMIN_WERK_FK',right_on='ID')


# In[681]:


df_bnd_werk['BUENDEL_FK'].nunique()


# In[682]:


df_bnd_werk.columns


# In[683]:


df_bnd_werk_drop = df_bnd_werk[['BUENDEL_FK','WERKSKENNBUCHSTABE']].drop_duplicates(subset=None,keep='first')


# In[684]:


df_amount_werk = df_bnd_werk_drop.groupby(['BUENDEL_FK'])['WERKSKENNBUCHSTABE'].nunique().reset_index()
df_amount_werk.columns = ['BUENDEL_FK','ANZAHL_WERK']
df_amount_werk['ANZAHL_WERK'] = df_amount_werk['ANZAHL_WERK'].fillna(0)

df_bnd_werk_amount = df_bnd_werk_drop.merge(df_amount_werk,left_on='BUENDEL_FK',right_on='BUENDEL_FK')


# In[685]:


df_bnd_werk_amount['WERKSKENNBUCHSTABE'].unique()


# In[686]:


df_bnd_werk_noduplicate = df_bnd_werk_amount.groupby(['BUENDEL_FK'])['WERKSKENNBUCHSTABE'].apply(';'.join).reset_index()
df_bnd_werk_noduplicate = df_bnd_werk_noduplicate.merge(df_bnd_werk_amount,left_on='BUENDEL_FK',right_on='BUENDEL_FK')
df_bnd_werk_noduplicate = df_bnd_werk_noduplicate[['BUENDEL_FK','WERKSKENNBUCHSTABE_x','ANZAHL_WERK']].drop_duplicates(subset=None, keep='first')
df_bnd_werk_noduplicate.columns = ['BUENDEL_FK','WERKSKENNBUCHSTABE','ANZAHL_WERK']

df_bnd_werk_noduplicate


# In[687]:


# Merge Werk information with Bündels
new_df_bnd_drop_produkt_werk = new_df_bnd_drop_produkt.merge(df_bnd_werk_noduplicate, 
                                                             left_on='Bnd_ID',
                                                             right_on='BUENDEL_FK',how='left')


# In[688]:


new_df_bnd_drop_produkt_werk.columns


# In[689]:


new_df_bnd_drop_produkt_werk['BUENDEL_NUMMER'].nunique()


# In[690]:


new_df_bnd_drop_produkt_werk[new_df_bnd_drop_produkt_werk['WERKSKENNBUCHSTABE'].isnull()]['BUENDEL_NUMMER'].nunique()


# #### Add affected external partners to Bündels

# In[691]:


df_ctime_terminated = pd.read_parquet(home_dir_tables + 'T_TXT_CTIME_TERMINATED.parquet', engine='pyarrow')

df_ctime_terminated_selected = df_ctime_terminated[['BUENDELNUMMER','LIEFERANTENNUMMER','LIEFERANTENNAME','WERK']].drop_duplicates(subset=None, keep='first')
df_ctime_terminated_selected.columns = ['BUENDEL_NUMMER','LIEFERANTEN_NUMMER','LIEFERANTEN_NAME','WERK_Ctime']


# In[692]:


df_ctime_terminated_selected


# In[693]:


df_ctime_terminated_selected.nunique()


# In[694]:


df_ctime_terminated_selected[df_ctime_terminated_selected['BUENDEL_NUMMER']=='0039551-001']['LIEFERANTEN_NUMMER'].unique()


# In[695]:


df_ctime_terminated_selected[df_ctime_terminated_selected['BUENDEL_NUMMER']=='0039551-001']['WERK_Ctime'].unique()


# In[696]:


df_amount_EXT = df_ctime_terminated_selected.groupby(['BUENDEL_NUMMER'])['LIEFERANTEN_NUMMER'].nunique().reset_index()
df_amount_EXT.columns = ['BUENDEL_NUMMER','ANZAHL_EXT']

#add ext. amount to dataframe
df_EXT = df_ctime_terminated_selected.merge(df_amount_EXT,left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER')


# In[697]:


df_EXT['ANZAHL_EXT'].describe()


# In[698]:


df_EXT


# In[699]:


df_amount_werk = df_ctime_terminated_selected.groupby(['BUENDEL_NUMMER'])['WERK_Ctime'].nunique().reset_index()
df_amount_werk.columns = ['BUENDEL_NUMMER','ANZAHL_WERK_Ctime']

#add werk(ctime) amount to dataframe
df_EXT = df_EXT.merge(df_amount_werk,left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER')


# In[700]:


df_EXT['LIEFERANTEN_NUMMER'] = df_EXT['LIEFERANTEN_NUMMER'].fillna('None')
df_EXT['LIEFERANTEN_NAME'] = df_EXT['LIEFERANTEN_NAME'].fillna('None')
df_EXT['WERK_Ctime'] = df_EXT['WERK_Ctime'].fillna('None')
df_EXT['ANZAHL_EXT'] = df_EXT['ANZAHL_EXT'].fillna(0)
df_EXT['ANZAHL_WERK_Ctime'] = df_EXT['ANZAHL_WERK_Ctime'].fillna(0)


# In[701]:


df_EXT.dtypes


# In[702]:


df_EXT_nummer = df_EXT[['BUENDEL_NUMMER','LIEFERANTEN_NUMMER']].drop_duplicates()
concat_df_EXT_nummer = df_EXT_nummer.groupby(['BUENDEL_NUMMER'])['LIEFERANTEN_NUMMER'].apply(';'.join).reset_index()

df_EXT_name = df_EXT[['BUENDEL_NUMMER','LIEFERANTEN_NAME']].drop_duplicates()
concat_df_EXT_name = df_EXT_name.groupby(['BUENDEL_NUMMER'])['LIEFERANTEN_NAME'].apply(';'.join).reset_index()

df_EXT_werk = df_EXT[['BUENDEL_NUMMER','WERK_Ctime']].drop_duplicates()
concat_df_EXT_werk = df_EXT_werk.groupby(['BUENDEL_NUMMER'])['WERK_Ctime'].apply(';'.join).reset_index()


df_EXT_info = df_EXT.merge(concat_df_EXT_nummer,left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER')
df_EXT_info = df_EXT_info.merge(concat_df_EXT_name, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER')
df_EXT_info = df_EXT_info.merge(concat_df_EXT_werk, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER')


# In[703]:


df_EXT_info


# In[704]:


df_EXT_info = df_EXT_info[['BUENDEL_NUMMER','LIEFERANTEN_NUMMER_y','LIEFERANTEN_NAME_y',
                           'WERK_Ctime_y','ANZAHL_EXT','ANZAHL_WERK_Ctime']].drop_duplicates(subset=None, keep='first')

df_EXT_info.columns = ['BUENDEL_NUMMER','LIEFERANTEN_NUMMER','LIEFERANTEN_NAME',
                      'WERK_Ctime','ANZAHL_EXT','ANZAHL_WERK_Ctime']


# In[705]:


df_EXT_info


# In[706]:


new_df_bnd_drop_produkt_werk_ext = new_df_bnd_drop_produkt_werk.merge(df_EXT_info, 
                                                             left_on='BUENDEL_NUMMER',
                                                             right_on='BUENDEL_NUMMER',how='left')


# In[707]:


new_df_bnd_drop_produkt_werk_ext.head(5)


# In[708]:


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

# In[709]:


#new_df_bnd_produkt_ab['BR'].nunique()


# In[710]:



new_df_bnd_produkt_ab = new_df_bnd_drop_produkt_werk_ext.loc[new_df_bnd_drop_produkt_werk['new_ProduktKlasse'].str.contains('A- B-Klasse')]
#new_df_bnd_produkt_ab = new_df_bnd_drop_produkt_werk_ext.loc[new_df_bnd_drop_produkt_werk['new_ProduktKlasse'].str.contains('C-Klasse')]
new_df_bnd_produkt_ab


# In[711]:


# for bundels which have no IST/SOLl Zustand, use the IST/SOLl Zustand from EC
new_df_bnd_produkt_ab['new_IST'] = new_df_bnd_produkt_ab.apply(new_IST, axis=1)
new_df_bnd_produkt_ab['new_SOLL'] = new_df_bnd_produkt_ab.apply(new_SOLL, axis=1)


# In[712]:


new_df_bnd_produkt_ab['ANZAHL_EXT'].describe()


# In[713]:


new_df_bnd_produkt_ab.columns


# In[714]:


# choose relevant columns to build Bündel Dataframe
df_bnd_ab = new_df_bnd_produkt_ab[['Bnd_ID', 'BUENDEL_NUMMER','Bnd_CREATION_DATE','Bnd_RUECKMELDE_DATUM', 'Bnd_MODIFICATION_DATE',
                                 'Bnd_STATUS','Bnd_BENENNUNG','EC_STICHWORT','EC_VERURSACHER', 'KAT_BENENNUNG','new_IST','new_SOLL',
                                 'Bnd_GEWUENSCHTER_EINSATZ','Bnd_KOSTEN_RELEVANT','Bnd_ZERTIFIZIERUNGS_RELEVANT',
                                 'Bnd_PROZESS_STATUS','Bnd_VERANTWORTLICHER_FK','Bnd_GREMIUM_ENTSCHEIDUNG_MANUELL',
                                 'Bnd_STATUS_GENEHMIGUNG','Bnd_RUECKSPRUNG_KOMMENTAR','Bnd_KOSTENBEWERTUNGS_ART',
                                 'Bnd_BEGR_RUECKSPRUNG','Bnd_MODULKOMPONENTE','MK_BENENNUNG_y','new_ProduktKlasse',
                                 'BR','AA',
                                 'WERKSKENNBUCHSTABE','LIEFERANTEN_NUMMER','LIEFERANTEN_NAME', 'WERK_Ctime',
                                 'ANZAHL_PAKET','ANZAHL_WERK','ANZAHL_EXT','ANZAHL_WERK_Ctime']].drop_duplicates(subset=None, keep='first')


# In[715]:


df_bnd_ab.columns = ['Bnd_ID', 'BUENDEL_NUMMER', 'CREATION_DATE', 'RUECKMELDE_DATUM','MODIFICATION_DATE',
                     'STATUS', 'BENENNUNG', 'STICHWORT','VERURSACHER','KAT', 'new_IST','new_SOLL', 
                     'GEWUENSCHTER_EINSATZ','KOSTEN_RELEVANT','ZERTIFIZIERUNGS_RELEVANT',
                     'PROZESS_STATUS','VERANTWORTLICHER_FK', 'GREMIUM_ENTSCHEIDUNG_MANUELL', 
                     'STATUS_GENEHMIGUNG','RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART', 
                     'BEGR_RUECKSPRUNG', 'MK', 'MK_BENENNUNG', 'ProduktKlasse', 
                     'BR','AA',
                     'WERKSKENNBUCHSTABE','LIEFERANTEN_NUMMER','LIEFERANTEN_NAME', 'WERK_Ctime',
                     'ANZAHL_PAKET','ANZAHL_WERK','ANZAHL_EXT','ANZAHL_WERK_Ctime']


# In[716]:


df_bnd_ab['BRAA'] = df_bnd_ab[['BR', 'AA']].apply(lambda x: '/'.join(x), axis=1)


# In[717]:


df_bnd_ab[['BR','AA','BRAA']]


# In[718]:


df_bnd_ab['BR'].nunique()


# In[719]:


df_bnd_ab['ZERTIFIZIERUNGS_RELEVANT'].unique()


# In[720]:


df_bnd_ab.groupby(['KOSTENBEWERTUNGS_ART'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# ### Remove test ECs

# In[721]:


# test ECs
indexNames_testecs = df_bnd_ab.loc[(df_bnd_ab['BENENNUNG'].isin(['test','Test'])) 
                   | (df_bnd_ab['STICHWORT'].isin(['test','Test'])) 
                   | (df_bnd_ab['new_IST'].isin(['test','Test','Siehe Anhang','siehe jeweiliges Bündel']))
                   | (df_bnd_ab['new_SOLL'].isin(['test','Test','Siehe Anhang','siehe jeweiliges Bündel']))].index.tolist()
indexNames_testecs


# In[722]:


df_bnd_ab.drop(indexNames_testecs,inplace=True)


# In[723]:


# Count word length in each description
df_bnd_ab['word_count_benennung'] = df_bnd_ab['BENENNUNG'].apply(lambda x: len(str(x).split(" ")))
df_bnd_ab['word_count_stichwort'] = df_bnd_ab['STICHWORT'].apply(lambda x: len(str(x).split(" ")))
df_bnd_ab['word_count_rueck'] = df_bnd_ab['RUECKSPRUNG_KOMMENTAR'].apply(lambda x: len(str(x).split(" ")))
df_bnd_ab['word_count_berg_rueck'] = df_bnd_ab['BEGR_RUECKSPRUNG'].apply(lambda x: len(str(x).split(" ")))
df_bnd_ab['word_count_ist'] = df_bnd_ab['new_IST'].apply(lambda x: len(str(x).split(" ")))
df_bnd_ab['word_count_soll'] = df_bnd_ab['new_SOLL'].apply(lambda x: len(str(x).split(" ")))


# In[724]:


df_bnd_ab


# In[725]:


df_buendel_kat_ab=df_bnd_ab[['BUENDEL_NUMMER','KAT']].drop_duplicates()
df_buendel_kat_ab['KAT'] = df_buendel_kat_ab['KAT'].astype('str')
concat_df_buendel_kat_ab = df_buendel_kat_ab.groupby(['BUENDEL_NUMMER'])['KAT'].apply(','.join).reset_index()

# sort strings in column EC_KATEGORIE of df_x
for i, row in concat_df_buendel_kat_ab.iterrows():
    tmp_cat = list(row['KAT'].split(sep=','))
    tmp_cat.sort()
    cat = ','.join([str(elem) for elem in tmp_cat]) 
    #temp_art = ' '.join(set(row['ProduktArt'].split(sep=',')))
    concat_df_buendel_kat_ab.at[i,'KAT'] = cat

concat_df_buendel_kat_ab


# In[726]:


df_buendel_braa_ab = df_bnd_ab[['BUENDEL_NUMMER','BRAA']].drop_duplicates()
df_buendel_braa_ab['BRAA'] = df_buendel_braa_ab['BRAA'].astype('str')

concat_df_buendel_braa_ab = df_buendel_braa_ab.groupby(['BUENDEL_NUMMER'])['BRAA'].apply(';'.join).reset_index()


# In[727]:


concat_df_buendel_braa_ab


# In[728]:


df_bnd_ab = df_bnd_ab.merge(concat_df_buendel_kat_ab, left_on='BUENDEL_NUMMER', right_on='BUENDEL_NUMMER')
df_bnd_ab = df_bnd_ab.merge(concat_df_buendel_braa_ab, left_on='BUENDEL_NUMMER', right_on='BUENDEL_NUMMER')


# In[729]:


df_bnd_ab[['ANZAHL_PAKET','BRAA_y']].head(50)


# In[730]:


df_bnd_ab = df_bnd_ab.rename(columns={'BRAA_y':'BRAA'})
df_bnd_ab = df_bnd_ab.drop(columns=['BRAA_x'])


# In[731]:


df_bnd_ab.columns


# In[732]:


df_bnd_ab[['ANZAHL_PAKET','BRAA']]


# In[733]:


df_bnd_ab = df_bnd_ab.rename(columns={'KAT_x':'KAT'})
df_bnd_ab = df_bnd_ab.drop(columns=['KAT_y'])


# In[734]:


df_bnd_ab = df_bnd_ab.drop_duplicates(subset=None, keep='first')
df_bnd_ab.head()


# ## 3. Analyzing Rücksprung Kommentar - frequent words (unigram/bigram/trigram)

# In[735]:


df_bnd_ab.loc[df_bnd_ab['RUECKSPRUNG_KOMMENTAR'].str.contains('NaN')==False]['BUENDEL_NUMMER'].nunique()


# ### Preprocessing 

# In[736]:


list_ruck_str = []
list_ruck_str = df_bnd_ab.loc[df_bnd_ab['RUECKSPRUNG_KOMMENTAR'].str.contains('NaN')==False]['RUECKSPRUNG_KOMMENTAR'].values.tolist()
list_ruck_str


# In[737]:


from nltk.stem.snowball import GermanStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = SnowballStemmer('german')

stops = set(stopwords.words('german'))
stops.update(set(stopwords.words('english')))

stops.update(['bündel','wurde','wurden','aufgrund','hinzu','bitte','wegen','siehe','rücksprung','bereits','and'])
  

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


# In[738]:


docs = []
docs = docs_preprocessor(list_ruck_str)
docs


# ### Most frequent words in Rucksprung Kommentar

# In[739]:


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


# In[740]:


fdist.N()


# In[741]:


fdist.most_common(50)


# ### Creating a vector of word counts 

# In[742]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_df=0.8, stop_words=stops, max_features=10000, ngram_range=(1,3))


# In[743]:


import string
str_docs = []
for i in range(len(docs)):
    str_docs.append([])   # appending a new list!

for idx in range(len(docs)):
    str_docs[idx] = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in docs[idx]]).strip()

str_docs[1]


# In[744]:


X = cv.fit_transform(str_docs)


# In[745]:


list(cv.vocabulary_.keys())[:10]


# In[746]:


#Most frequently occuring words
def get_top_n_words(docs, n=None):
    vec = CountVectorizer(min_df=3).fit(docs)
    bag_of_words = vec.transform(docs)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]


# In[747]:


#Convert most freq words to datagrame for plotting bar plot
top_words = get_top_n_words(str_docs, n=30)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word","Freq"]

#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(20,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)


# In[748]:


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


# In[749]:


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


# In[750]:


index_ruck_snr = df_bnd_ab.loc[df_bnd_ab['RUECKSPRUNG_KOMMENTAR'].str.contains('SNR')]['RUECKSPRUNG_KOMMENTAR'].index.tolist()
index_ruck_sachnummer = df_bnd_ab.loc[df_bnd_ab['RUECKSPRUNG_KOMMENTAR'].str.contains('Sachnummer')]['RUECKSPRUNG_KOMMENTAR'].index.tolist()
index_ruck_teil = df_bnd_ab.loc[df_bnd_ab['RUECKSPRUNG_KOMMENTAR'].str.contains('Teil')]['RUECKSPRUNG_KOMMENTAR'].index.tolist()


# In[751]:


list_ruck_snr_relevant = []
list_ruck_snr_relevant = list(set(index_ruck_snr + index_ruck_sachnummer + index_ruck_teil))
len(list_ruck_snr_relevant)


# ## 4. Analyzing influencing factors of processing time of Bündels

# In[752]:


df_bnd_ab.columns


# In[753]:


# Focus on Abgeschlossen EC
df_bnd_ab_abgeschlossen = df_bnd_ab.loc[df_bnd_ab['PROZESS_STATUS'] == 'ABGESCHLOSSEN']


# In[754]:


df_bnd_ab['BUENDEL_NUMMER'].nunique()


# In[755]:


df_bnd_ab_abgeschlossen.columns


# In[756]:


# for S-Class, there are 24462 Bündels
# Focus on 'Abgeschlossen' Bündels
df_bnd_ab.groupby(['PROZESS_STATUS'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[757]:


df_bnd_ab_abgeschlossen.groupby(['RUECKMELDE_DATUM'])['BUENDEL_NUMMER'].nunique()


# In[758]:


list_rueckdatum = df_bnd_ab_abgeschlossen['RUECKMELDE_DATUM'].tolist()


# In[759]:


list_rueckdatum.sort()


# In[760]:


# Feedback date is from 2014 to 2020
list_rueckdatum


# In[761]:


# Calculate duration of processing time of finished ECs
df_bnd_ab_abgeschlossen['CREATION_DATE'] = df_bnd_ab_abgeschlossen['CREATION_DATE'].apply(lambda x: pd.to_datetime(x).date())
df_bnd_ab_abgeschlossen['RUECKMELDE_DATUM'] = df_bnd_ab_abgeschlossen['RUECKMELDE_DATUM'].apply(lambda x: pd.to_datetime(x).date())
#df_bnd_ab_abgeschlossen['MODIFICATION_DATE'] = df_bnd_s_abgeschlossen['MODIFICATION_DATE'].apply(lambda x: pd.to_datetime(x).date())

df_bnd_ab_abgeschlossen['DAUER'] = df_bnd_ab_abgeschlossen['RUECKMELDE_DATUM'] - df_bnd_ab_abgeschlossen['CREATION_DATE']


# In[762]:


df_bnd_ab_abgeschlossen[['CREATION_DATE','RUECKMELDE_DATUM','DAUER']]


# In[763]:


df_bnd_ab_abgeschlossen['DAUER'].sort_values(ascending=False)


# In[764]:


df_bnd_ab_abgeschlossen_dauer = df_bnd_ab_abgeschlossen.loc[(df_bnd_ab_abgeschlossen['DAUER'] > '1000 days')==True]
df_bnd_ab_abgeschlossen_dauer


# In[765]:


df_bnd_ab_abgeschlossen_dauer[['BUENDEL_NUMMER','ANZAHL_PAKET','BRAA']]


# In[766]:


# drop bundels with wrong Ruckmeldung_datum, e.g. in year 2033...2099 
df_bnd_ab_abgeschlossen_dauer = df_bnd_ab_abgeschlossen.loc[(df_bnd_ab_abgeschlossen['DAUER'] > '1000 days')==False]
df_bnd_ab_abgeschlossen_dauer


# In[767]:


# average duration of Bundel processing time for S-Class is 66 days
df_bnd_ab_abgeschlossen_dauer['DAUER'].describe()


# In[768]:


df_bnd_ab_abgeschlossen_dauer.loc[df_bnd_ab_abgeschlossen_dauer['DAUER']<'1 days 00:00:00']


# In[769]:


# Histogram Dauer
quantile_list = [0, .25, .5, .75, 1.]


quantiles = df_bnd_ab_abgeschlossen_dauer['DAUER'].quantile(quantile_list).astype('timedelta64[D]').quantile(quantile_list)
quantiles

# quantiles_new = df_bnd_s_abgeschlossen_dauer.loc[input_df_cal_dauer['RUECKSPRUNG_KOMMENTAR'].str.contains('nan')==False]['Dauer'].astype('timedelta64[D]').quantile(quantile_list)
# quantiles_new


# In[770]:


fig, ax = plt.subplots(figsize=(15,8))
df_bnd_ab_abgeschlossen_dauer['DAUER'].astype('timedelta64[D]').hist(bins=70, color='#A9C5D3', 
                             edgecolor='black', grid=False)
for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)
ax.set_title('Distribution of ECR lead time"', 
             fontsize=12)
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('Number of ECRs', fontsize=12)


# In[771]:


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
    if row['DAUER'] <= pd.to_timedelta('41 days'):
        val = 'Shorter'
    #elif (row['DAUER'] > pd.to_timedelta('42 days')) & (row['DAUER'] <= pd.to_timedelta('64 days')):
        #val = 'Q2'
#     elif (row['DAUER'] > pd.to_timedelta('33 days')) & (row['DAUER'] <= pd.to_timedelta('64 days')):
#         val = 'Q3'
    else:
        val = 'Longer'
    return val

df_bnd_ab_abgeschlossen_dauer['DAUER_KAT'] = df_bnd_ab_abgeschlossen_dauer.apply(dauer_cat, axis=1)


# In[772]:


df_bnd_ab_abgeschlossen_dauer[['DAUER','DAUER_KAT']]


# ## 5. Change Impact Analysis

# ### Load SNR related data

# In[773]:


df_snrimbuendel = pd.read_parquet(home_dir_tables + 'T_ACM_EKCRBUSD301_SachnummerImBuendelEntity.parquet', engine='pyarrow')


# In[774]:


df_snrimbuendel = df_snrimbuendel[['SACHNUMMER','BUENDEL_FK','MODULSUBKOMPONENTE','ZB_BENENNUNG',
                                   'MODULGRUPPE','EMISSIONSKENNZEICHEN','BESCHAFFUNGSART',
                                   'SICHERHEITSRELEVANT','TEILEART','WERKZEUGAENDERUNG_ERFORDERLICH',
                                   'EE_STEUERGERAET_BETROFFEN','BETRIEBSANLEITUNG_BETROFFEN',
                                   'LEITUNGSSATZ_BETROFFEN','LASTENHEFTAENDERUNG_ERFORDERLICH',
                                   'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']]


# In[775]:


# merge with buendel entity to get buendel number
new_df_linkbuendelsnr = df_bnd_ab_abgeschlossen_dauer.merge(df_snrimbuendel, left_on='Bnd_ID', right_on='BUENDEL_FK')


# In[776]:


new_df_linkbuendelsnr.head()


# In[777]:


# choose relavent columns to form a new dataframe
df_linkbuendelsnr = pd.DataFrame
df_linkbuendelsnr = new_df_linkbuendelsnr[['BUENDEL_NUMMER','SACHNUMMER','MODULSUBKOMPONENTE','ZB_BENENNUNG','MODULGRUPPE',
                                          'EMISSIONSKENNZEICHEN','BESCHAFFUNGSART',
                                          'SICHERHEITSRELEVANT','TEILEART','WERKZEUGAENDERUNG_ERFORDERLICH',
                                          'EE_STEUERGERAET_BETROFFEN','BETRIEBSANLEITUNG_BETROFFEN',
                                          'LEITUNGSSATZ_BETROFFEN','LASTENHEFTAENDERUNG_ERFORDERLICH',
                                          'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']]
df_linkbuendelsnr


# In[778]:


# only focus on A-SNR
df_linkbuendelsnr['Kennbuchstabe'] = df_linkbuendelsnr['SACHNUMMER'].astype(str).str[0]
indexNames_dropsnr_notA = df_linkbuendelsnr.loc[df_linkbuendelsnr['Kennbuchstabe'] != 'A'].index
df_linkbuendelsnr.drop(indexNames_dropsnr_notA, inplace=True)


# In[779]:


df_linkbuendelsnr['MODULSUBKOMPONENTE'] = df_linkbuendelsnr['MODULSUBKOMPONENTE'].fillna(0)
df_linkbuendelsnr['MODULGRUPPE'] = df_linkbuendelsnr['MODULGRUPPE'].fillna(999)

# convert columns to appropriate datatype
df_linkbuendelsnr['BUENDEL_NUMMER'] = df_linkbuendelsnr['BUENDEL_NUMMER'].astype('str')
df_linkbuendelsnr['SACHNUMMER'] = df_linkbuendelsnr['SACHNUMMER'].astype('str')
df_linkbuendelsnr['MODULSUBKOMPONENTE'] = df_linkbuendelsnr['MODULSUBKOMPONENTE'].astype('int')
df_linkbuendelsnr['ZB_BENENNUNG'] = df_linkbuendelsnr['ZB_BENENNUNG'].astype('str')
df_linkbuendelsnr['MODULGRUPPE'] = df_linkbuendelsnr['MODULGRUPPE'].astype('int')


# In[780]:


# extract MK und MSK from the 5-digit MODULSUBKOMPONENT
for i, row in df_linkbuendelsnr.iterrows():
    temp_msk = str(row['MODULSUBKOMPONENTE']).zfill(5)
    #df_linkbuendelsnr.set_value(i,'MSK',temp_msk)
    df_linkbuendelsnr.at[i,'MSK'] = temp_msk


# In[781]:


for i, row in df_linkbuendelsnr.iterrows():
    temp_mk = row['MSK'][0:3]
    temp_msubk = row['MSK'][3:5]
    #df_linkbuendelsnr.set_value(i,'MK',temp_mk)
    #df_linkbuendelsnr.set_value(i,'SUBK',temp_msubk)
    df_linkbuendelsnr.at[i,'MK'] = temp_mk
    df_linkbuendelsnr.at[i,'SUBK'] = temp_msubk


# In[782]:


# drop items in which MG or MK is invalid
df_linkbuendelsnr = df_linkbuendelsnr.drop('MODULSUBKOMPONENTE', 1)


# ### Decompose SNR into:
# 
# * Kennbuchstabe (SNR-Kennbuchstabe) → 1. Stelle in der A-Sachnummer
# * Typzahl → 2. bis einschließlich 4. Stelle in der A-Sachnummer
# * Konstruktions_Haupt_und_Untergruppe (Konstruktions-Haupt- und Untergruppe) → 5. bis einschließlich 7. Stelle der A-Sachnummer
# * Fortlaufende_Nummer (Abwandlung oder fortlaufende Nummer) → 8. und 9. Stelle der A-Sachnummer
# * Teilnummer_Untergruppe (Teilnummer bezogen auf die Untergruppe) → 10. und 11. Stelle der A-Sachnummer

# In[783]:


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

# In[784]:


df_linkbuendelsnr.head()


# In[1429]:


df_linkbuendelsnr[df_linkbuendelsnr['BUENDEL_NUMMER']=='0053111-001']['SACHNUMMER']


# In[785]:


df_linkbuendelsnr['BUENDEL_NUMMER'].nunique()


# In[786]:


df_amount_snr = df_linkbuendelsnr.groupby(['BUENDEL_NUMMER'])['SACHNUMMER'].nunique().reset_index()
df_amount_snr.columns = ['BUENDEL_NUMMER','ANZAHL_SACHNUMMER']

df_amount_snr_benennung = df_linkbuendelsnr.groupby(['BUENDEL_NUMMER'])['ZB_BENENNUNG'].nunique().reset_index()
df_amount_snr_benennung.columns = ['BUENDEL_NUMMER','ANZAHL_ZB_BENENNUNG']

df_linkbuendelsnr = df_linkbuendelsnr.merge(df_amount_snr, left_on = 'BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
df_linkbuendelsnr = df_linkbuendelsnr.merge(df_amount_snr_benennung, left_on = 'BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[787]:


df_buendel_linked = df_linkbuendelsnr.merge(df_bnd_ab_abgeschlossen_dauer, left_on = 'BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[788]:


df_buendel_linked = df_buendel_linked.drop_duplicates()


# In[789]:


df_buendel_linked.columns


# In[790]:


df_buendel_linked = df_buendel_linked.drop(columns=['BR','AA'])


# In[791]:


df_buendel_linked.columns = ['BUENDEL_NUMMER', 'SACHNUMMER', 'ZB_BENENNUNG', 'MODULGRUPPE',
                             'EMISSIONSKENNZEICHEN', 'BESCHAFFUNGSART', 'SICHERHEITSRELEVANT',
                             'TEILEART', 'WERKZEUGAENDERUNG_ERFORDERLICH',
                             'EE_STEUERGERAET_BETROFFEN', 'BETRIEBSANLEITUNG_BETROFFEN',
                             'LEITUNGSSATZ_BETROFFEN', 'LASTENHEFTAENDERUNG_ERFORDERLICH',
                             'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN','Kennbuchstabe', 'MSK', 'MK', 
                             'SUBK', 'TypZahl', 'KG', 'U', 'KGU','Fortlaufende_Nummer', 
                             'Teilnummer_Untergruppe', 'ANZAHL_SACHNUMMER', 'ANZAHL_ZB_BENENNUNG',
                             'Bnd_ID','CREATION_DATE', 'RUECKMELDE_DATUM', 'MODIFICATION_DATE',
                             'STATUS', 'BENENNUNG', 'STICHWORT','VERURSACHER','Bnd_KAT','new_IST', 
                             'new_SOLL', 'GEWUENSCHTER_EINSATZ', 'KOSTEN_RELEVANT',
                             'ZERTIFIZIERUNGS_RELEVANT', 'PROZESS_STATUS', 'VERANTWORTLICHER_FK',
                             'GREMIUM_ENTSCHEIDUNG_MANUELL', 'STATUS_GENEHMIGUNG',
                             'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART', 'BEGR_RUECKSPRUNG',
                             'Bnd_MK', 'Bnd_MK_BENENNUNG', 'ProduktKlasse', 'WERKSKENNBUCHSTABE',
                             'LIEFERANTEN_NUMMER', 'LIEFERANTEN_NAME', 'WERK_Ctime','ANZAHL_PAKET', 
                             'ANZAHL_WERK', 'ANZAHL_EXT', 'ANZAHL_WERK_Ctime',
                             'word_count_benennung','word_count_stichwort', 'word_count_rueck', 
                             'word_count_berg_rueck','word_count_ist', 'word_count_soll',  
                             'BRAA','DAUER', 'DAUER_KAT']


# In[792]:


df_buendel_linked = df_buendel_linked.merge(df_mk, left_on='MK', right_on='MK')


# In[793]:


df_buendel_linked[['BUENDEL_NUMMER','ANZAHL_PAKET','BRAA']]


# In[794]:


df_buendel_linked.loc[df_buendel_linked['ANZAHL_PAKET']==2]['BRAA'].head(200)


# In[795]:


# add prefix for MK/KG for better understanding
MK_arr = []
KG_arr = []
MK_arr = ['MK_'+ s for s in df_buendel_linked['MK'].values.astype(str)]
KG_arr = ['KG_'+ s for s in df_buendel_linked['KG'].values.astype(str)]

df_buendel_linked['MK'] = MK_arr
df_buendel_linked['KG'] = KG_arr


# In[796]:


#df_buendel_linked[['ANZAHL_PAKET','BRAA']].iloc[462772]


# #### Combine multi variables into one.

# In[797]:


df_buendel_typ = df_buendel_linked[['BUENDEL_NUMMER','TypZahl']].drop_duplicates()
concat_df_buendel_typ = df_buendel_typ.groupby(['BUENDEL_NUMMER'])['TypZahl'].apply(';'.join).reset_index()
concat_df_buendel_typ


# In[798]:


df_buendel_mk = df_buendel_linked[['BUENDEL_NUMMER','MK_BENENNUNG']].drop_duplicates()
concat_df_buendel_mk = df_buendel_mk.groupby(['BUENDEL_NUMMER'])['MK_BENENNUNG'].apply(';'.join).reset_index()
concat_df_buendel_mk


# In[799]:


df_buendel_subk = df_buendel_linked[['BUENDEL_NUMMER','SUBK']].drop_duplicates()
concat_df_buendel_subk = df_buendel_subk.groupby(['BUENDEL_NUMMER'])['SUBK'].apply(';'.join).reset_index()
concat_df_buendel_subk


# In[800]:


df_buendel_msk = df_buendel_linked[['BUENDEL_NUMMER','MSK']].drop_duplicates()
concat_df_buendel_msk = df_buendel_msk.groupby(['BUENDEL_NUMMER'])['MSK'].apply(';'.join).reset_index()
concat_df_buendel_msk


# In[801]:


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


# In[802]:


concat_df_buendel_kat['Bnd_KAT'].nunique()


# In[803]:


df_buendel_kg = df_buendel_linked[['BUENDEL_NUMMER','KG']].drop_duplicates()
df_buendel_kg['KG'] = df_buendel_kg['KG'].astype('str')
concat_df_buendel_kg = df_buendel_kg.groupby(['BUENDEL_NUMMER'])['KG'].apply(';'.join).reset_index()
concat_df_buendel_kg


# In[804]:


df_buendel_u = df_buendel_linked[['BUENDEL_NUMMER','U']].drop_duplicates()
df_buendel_u['U'] = df_buendel_u['U'].astype('str')
concat_df_buendel_u = df_buendel_u.groupby(['BUENDEL_NUMMER'])['U'].apply(';'.join).reset_index()
concat_df_buendel_u


# In[805]:


df_buendel_kgu = df_buendel_linked[['BUENDEL_NUMMER','KGU']].drop_duplicates()
df_buendel_kgu['KGU'] = df_buendel_kgu['KGU'].astype('str')
concat_df_buendel_kgu = df_buendel_kgu.groupby(['BUENDEL_NUMMER'])['KGU'].apply(';'.join).reset_index()
concat_df_buendel_kgu


# In[806]:


# Emissionsart
df_buendel_emi = df_buendel_linked[['BUENDEL_NUMMER','EMISSIONSKENNZEICHEN']].drop_duplicates()
df_buendel_emi['EMISSIONSKENNZEICHEN'] = df_buendel_emi['EMISSIONSKENNZEICHEN'].astype('str')
concat_df_buendel_emi = df_buendel_emi.groupby(['BUENDEL_NUMMER'])['EMISSIONSKENNZEICHEN'].apply(';'.join).reset_index()
concat_df_buendel_emi


# In[807]:


concat_df_buendel_emi.groupby(['EMISSIONSKENNZEICHEN'])['BUENDEL_NUMMER'].nunique()


# In[808]:


def emission (s):
    tmp_emi = ""
    if (s == 'JA') | (s == 'JA;NEIN') | (s=='NEIN;JA'):
        tmp_emi = 'JA'
    else:
        tmp_emi = 'NEIN'
    return tmp_emi


# In[809]:


concat_df_buendel_emi['new_EMISSIONSKENNZEICHEN'] = concat_df_buendel_emi['EMISSIONSKENNZEICHEN'].apply(lambda x: emission(x))


# In[810]:


concat_df_buendel_emi.groupby(['new_EMISSIONSKENNZEICHEN'])['BUENDEL_NUMMER'].nunique()


# In[811]:


# Beschfungsart
df_buendel_beschafungsart = df_buendel_linked[['BUENDEL_NUMMER','BESCHAFFUNGSART']].drop_duplicates()
df_buendel_beschafungsart['BESCHAFFUNGSART'] = df_buendel_beschafungsart['BESCHAFFUNGSART'].astype('str')
concat_df_buendel_beschafung = df_buendel_beschafungsart.groupby(['BUENDEL_NUMMER'])['BESCHAFFUNGSART'].apply(';'.join).reset_index()
concat_df_buendel_beschafung


# In[812]:


concat_df_buendel_beschafung.groupby(['BESCHAFFUNGSART'])['BUENDEL_NUMMER'].nunique()


# In[813]:


def beschafung (s):
    tmp_besch = ""
    if 'BEIDES' in s:
        tmp_besch = 'BEIDES'
    elif (s == 'HAUSTEIL;KAUFTEIL') | (s == 'KAUFTEIL;HAUSTEIL'):
        tmp_besch = 'BEIDES'
    else:
        tmp_besch = s
    return tmp_besch


# In[814]:


concat_df_buendel_beschafung['new_BESCHAFFUNGSART'] = concat_df_buendel_beschafung['BESCHAFFUNGSART'].apply(lambda x: beschafung(x))
concat_df_buendel_beschafung.groupby(['new_BESCHAFFUNGSART'])['BUENDEL_NUMMER'].nunique()


# In[815]:


# Sicherheit
df_buendel_sicherheit = df_buendel_linked[['BUENDEL_NUMMER','SICHERHEITSRELEVANT']].drop_duplicates()
df_buendel_sicherheit['SICHERHEITSRELEVANT'] = df_buendel_sicherheit['SICHERHEITSRELEVANT'].astype('str')
concat_df_buendel_sicherheit = df_buendel_sicherheit.groupby(['BUENDEL_NUMMER'])['SICHERHEITSRELEVANT'].apply(';'.join).reset_index()
concat_df_buendel_sicherheit


# In[816]:


concat_df_buendel_sicherheit.groupby(['SICHERHEITSRELEVANT'])['BUENDEL_NUMMER'].nunique()


# In[817]:


def sicherheit (s):
    tmp_sicherheit = ""
    if 'JA' in s:
        tmp_sicherheit = 'JA'
    else:
        tmp_sicherheit = 'NEIN'
    return tmp_sicherheit


# In[818]:


concat_df_buendel_sicherheit['new_SICHERHEITSRELEVANT'] = concat_df_buendel_sicherheit['SICHERHEITSRELEVANT'].apply(lambda x: sicherheit(x))
concat_df_buendel_sicherheit.groupby(['new_SICHERHEITSRELEVANT'])['BUENDEL_NUMMER'].nunique()


# In[819]:


# Teileart
df_buendel_teileart = df_buendel_linked[['BUENDEL_NUMMER','TEILEART']].drop_duplicates()
df_buendel_teileart['TEILEART'] = df_buendel_teileart['TEILEART'].astype('str')
concat_df_buendel_teileart = df_buendel_teileart.groupby(['BUENDEL_NUMMER'])['TEILEART'].apply(';'.join).reset_index()
concat_df_buendel_teileart


# In[820]:


concat_df_buendel_teileart.groupby(['TEILEART'])['BUENDEL_NUMMER'].nunique()


# In[821]:


def teileart (s):
    tmp_teileart = ""
    if (s == "MO_TEIL;None") | (s == "None;MO_TEIL"):
        tmp_teileart = "MO_TEIL"
    elif (s == "AGGREGATE_TEIL;MO_TEIL") | (s== "MO_TEIL;AGGREGATE_TEIL"):
        tmp_teileart = "AGGREGATE_TEIL;MO_TEIL"
    elif (s == "AGGREGATE_TEIL;MO_TEIL;RB_TEIL") | (s == "AGGREGATE_TEIL;RB_TEIL;MO_TEIL") | (s == "MO_TEIL;AGGREGATE_TEIL;RB_TEIL") | (s == "MO_TEIL;RB_TEIL;AGGREGATE_TEIL") | (s == "RB_TEIL;AGGREGATE_TEIL;MO_TEIL"):
         tmp_teileart = "AGGREGATE_TEIL;MO_TEIL;RB_TEIL"
    elif (s == "AGGREGATE_TEIL;RB_TEIL") | (s == "RB_TEIL;AGGREGATE_TEIL"):
        tmp_teileart = "AGGREGATE_TEIL;RB_TEIL"
    elif (s == "MO_TEIL;RB_TEIL") | (s =="RB_TEIL;MO_TEIL"):
        tmp_teileart = "MO_TEIL;RB_TEIL"
    else:    
        tmp_teileart = s
    return tmp_teileart


# In[822]:


concat_df_buendel_teileart['new_TEILEART'] = concat_df_buendel_teileart['TEILEART'].apply(lambda x: teileart(x))
concat_df_buendel_teileart.groupby(['new_TEILEART'])['BUENDEL_NUMMER'].nunique()


# In[823]:


# WERKZEUGAENDERUNG_ERFORDERLICH
df_buendel_werkzeug = df_buendel_linked[['BUENDEL_NUMMER','WERKZEUGAENDERUNG_ERFORDERLICH']].drop_duplicates()
df_buendel_werkzeug['WERKZEUGAENDERUNG_ERFORDERLICH'] = df_buendel_werkzeug['WERKZEUGAENDERUNG_ERFORDERLICH'].astype('str')
concat_df_buendel_werkzeug = df_buendel_werkzeug.groupby(['BUENDEL_NUMMER'])['WERKZEUGAENDERUNG_ERFORDERLICH'].apply(';'.join).reset_index()
concat_df_buendel_werkzeug


# In[824]:


concat_df_buendel_werkzeug.groupby(['WERKZEUGAENDERUNG_ERFORDERLICH'])['BUENDEL_NUMMER'].nunique()


# In[825]:


def werkzeug (s):
    tmp_werkzeug = ""
    if 'JA' in s:
        tmp_werkzeug = 'JA'
    elif 'NEIN' in s:
        tmp_werkzeug = 'NEIN'
    else:
        tmp_werkzeug = 'None'
    return tmp_werkzeug


# In[826]:


concat_df_buendel_werkzeug['new_WERKZEUGAENDERUNG_ERFORDERLICH'] = concat_df_buendel_werkzeug['WERKZEUGAENDERUNG_ERFORDERLICH'].apply(lambda x: werkzeug(x))
concat_df_buendel_werkzeug.groupby(['new_WERKZEUGAENDERUNG_ERFORDERLICH'])['BUENDEL_NUMMER'].nunique()


# In[827]:


# EE_STEUERGERAET_BETROFFEN
df_buendel_ee = df_buendel_linked[['BUENDEL_NUMMER','EE_STEUERGERAET_BETROFFEN']].drop_duplicates()
df_buendel_ee['EE_STEUERGERAET_BETROFFEN'] = df_buendel_ee['EE_STEUERGERAET_BETROFFEN'].astype('str')
concat_df_buendel_ee = df_buendel_ee.groupby(['BUENDEL_NUMMER'])['EE_STEUERGERAET_BETROFFEN'].apply(';'.join).reset_index()
concat_df_buendel_ee


# In[828]:


concat_df_buendel_ee.groupby(['EE_STEUERGERAET_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[829]:


def ee(s):
    tmp_ee = ""
    if 'JA' in s:
        tmp_ee = 'JA'
    else:
        tmp_ee = 'NEIN'
    return tmp_ee


# In[830]:


concat_df_buendel_ee['new_EE_STEUERGERAET_BETROFFEN'] = concat_df_buendel_ee['EE_STEUERGERAET_BETROFFEN'].apply(lambda x: ee(x))
concat_df_buendel_ee.groupby(['new_EE_STEUERGERAET_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[831]:


# BETRIEBSANLEITUNG_BETROFFEN
df_buendel_anleitung = df_buendel_linked[['BUENDEL_NUMMER','BETRIEBSANLEITUNG_BETROFFEN']].drop_duplicates()
df_buendel_anleitung['BETRIEBSANLEITUNG_BETROFFEN'] = df_buendel_anleitung['BETRIEBSANLEITUNG_BETROFFEN'].astype('str')
concat_df_buendel_anleitung = df_buendel_anleitung.groupby(['BUENDEL_NUMMER'])['BETRIEBSANLEITUNG_BETROFFEN'].apply(';'.join).reset_index()
concat_df_buendel_anleitung


# In[832]:


concat_df_buendel_anleitung.groupby(['BETRIEBSANLEITUNG_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[833]:


def anleitung (s):
    tmp_anleitung = ""
    if 'JA' in s:
        tmp_anleitung = 'JA'
    else:
        tmp_anleitung = 'NEIN'
    return tmp_anleitung


# In[834]:


concat_df_buendel_anleitung['new_BETRIEBSANLEITUNG_BETROFFEN'] = concat_df_buendel_anleitung['BETRIEBSANLEITUNG_BETROFFEN'].apply(lambda x: anleitung(x))
concat_df_buendel_anleitung.groupby(['new_BETRIEBSANLEITUNG_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[835]:


# LEITUNGSSATZ_BETROFFEN
df_buendel_leitungssatz = df_buendel_linked[['BUENDEL_NUMMER','LEITUNGSSATZ_BETROFFEN']].drop_duplicates()
df_buendel_leitungssatz['LEITUNGSSATZ_BETROFFEN'] = df_buendel_leitungssatz['LEITUNGSSATZ_BETROFFEN'].astype('str')
concat_df_buendel_leitungssatz = df_buendel_leitungssatz.groupby(['BUENDEL_NUMMER'])['LEITUNGSSATZ_BETROFFEN'].apply(';'.join).reset_index()
concat_df_buendel_leitungssatz


# In[836]:


concat_df_buendel_leitungssatz.groupby(['LEITUNGSSATZ_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[837]:


def leitungssatz (s):
    tmp_ls = ""
    if 'JA' in s:
        tmp_ls = 'JA'
    else:
        tmp_ls = 'NEIN'
    return tmp_ls


# In[838]:


concat_df_buendel_leitungssatz['new_LEITUNGSSATZ_BETROFFEN'] = concat_df_buendel_leitungssatz['LEITUNGSSATZ_BETROFFEN'].apply(lambda x: leitungssatz(x))
concat_df_buendel_leitungssatz.groupby(['new_LEITUNGSSATZ_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[839]:


# LASTENHEFTAENDERUNG_ERFORDERLICH
df_buendel_lh = df_buendel_linked[['BUENDEL_NUMMER','LASTENHEFTAENDERUNG_ERFORDERLICH']].drop_duplicates()
df_buendel_lh['LASTENHEFTAENDERUNG_ERFORDERLICH'] = df_buendel_lh['LASTENHEFTAENDERUNG_ERFORDERLICH'].astype('str')
concat_df_buendel_lh = df_buendel_lh.groupby(['BUENDEL_NUMMER'])['LASTENHEFTAENDERUNG_ERFORDERLICH'].apply(';'.join).reset_index()
concat_df_buendel_lh


# In[840]:


concat_df_buendel_lh.groupby(['LASTENHEFTAENDERUNG_ERFORDERLICH'])['BUENDEL_NUMMER'].nunique()


# In[841]:



def lh (s):
    tmp_lh = ""
    if 'JA' in s:
        tmp_lh = 'JA'
    else:
        tmp_lh = 'NEIN'
    return tmp_lh


# In[842]:


concat_df_buendel_lh['new_LASTENHEFTAENDERUNG_ERFORDERLICH'] = concat_df_buendel_lh['LASTENHEFTAENDERUNG_ERFORDERLICH'].apply(lambda x: lh(x))
concat_df_buendel_lh.groupby(['new_LASTENHEFTAENDERUNG_ERFORDERLICH'])['BUENDEL_NUMMER'].nunique()


# In[843]:


# VERSUCHSTEIL_ASACHNUMMER_BETROFFEN
df_buendel_versuchteil = df_buendel_linked[['BUENDEL_NUMMER','VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']].drop_duplicates()
df_buendel_versuchteil['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'] = df_buendel_versuchteil['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'].astype('str')
concat_df_buendel_versuchteil = df_buendel_versuchteil.groupby(['BUENDEL_NUMMER'])['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'].apply(';'.join).reset_index()
concat_df_buendel_versuchteil


# In[844]:


concat_df_buendel_versuchteil.groupby(['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# In[845]:


def versuchteil (s):
    tmp_versuchteil = ""
    if 'JA' in s:
        tmp_versuchteil = 'JA'
    elif 'NEIN' in s:
        tmp_versuchteil = 'NEIN'
    else:
        tmp_versuchteil = 'None'
    return tmp_versuchteil


# In[846]:


concat_df_buendel_versuchteil['new_VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'] = concat_df_buendel_versuchteil['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'].apply(lambda x: versuchteil(x))
concat_df_buendel_versuchteil.groupby(['new_VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'])['BUENDEL_NUMMER'].nunique()


# ### Merge all variables via bnd-nummer

# In[847]:


concat_df_buendel_typ_mk = pd.merge(concat_df_buendel_typ, concat_df_buendel_mk, 
                                   left_on='BUENDEL_NUMMER', right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_typ_mk


# In[848]:


concat_df_buendel_mk_subk = pd.merge(concat_df_buendel_typ_mk, concat_df_buendel_subk, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_subk


# In[849]:


concat_df_buendel_mk_subk_msk = pd.merge(concat_df_buendel_mk_subk, concat_df_buendel_msk, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_subk_msk


# In[850]:


concat_df_buendel_mk_cat = pd.merge(concat_df_buendel_mk_subk_msk,concat_df_buendel_kat, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_cat


# In[851]:


concat_df_buendel_mk_cat_kg = pd.merge(concat_df_buendel_mk_cat, concat_df_buendel_kg, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_cat_kg


# In[852]:


concat_df_buendel_mk_cat_kg_u = pd.merge(concat_df_buendel_mk_cat_kg, concat_df_buendel_u, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_mk_cat_kg_u


# In[853]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_mk_cat_kg_u, concat_df_buendel_kgu, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
concat_df_buendel_linked


# In[854]:


df_amount_snr = df_buendel_linked.groupby(['BUENDEL_NUMMER'])['SACHNUMMER'].nunique().reset_index()
df_amount_snr.columns = ['BUENDEL_NUMMER','ANZAHL_SACHNUMMER']

df_amount_snr_benennung = df_buendel_linked.groupby(['BUENDEL_NUMMER'])['ZB_BENENNUNG'].nunique().reset_index()
df_amount_snr_benennung.columns = ['BUENDEL_NUMMER','ANZAHL_ZB_BENENNUNG']


# In[855]:


df_amount_snr


# In[856]:


df_amount_snr_benennung


# In[857]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_emi, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[858]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_beschafung, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[859]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_sicherheit, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[860]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_teileart, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[861]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_werkzeug, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[862]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_ee, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[863]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_anleitung, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[864]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_leitungssatz, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[865]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_lh, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[866]:


concat_df_buendel_linked = pd.merge(concat_df_buendel_linked, concat_df_buendel_versuchteil, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')


# In[867]:


concat_df_buendel_linked.columns


# In[868]:


concat_df_buendel_linked = concat_df_buendel_linked[['BUENDEL_NUMMER', 'TypZahl', 'MK_BENENNUNG', 'SUBK', 'MSK', 'Bnd_KAT',
       'KG', 'U', 'KGU', 
       'new_EMISSIONSKENNZEICHEN', 'new_BESCHAFFUNGSART', 'new_SICHERHEITSRELEVANT',
       'new_TEILEART', 'new_WERKZEUGAENDERUNG_ERFORDERLICH', 
       'new_EE_STEUERGERAET_BETROFFEN', 'new_BETRIEBSANLEITUNG_BETROFFEN',
       'new_LEITUNGSSATZ_BETROFFEN', 'new_LASTENHEFTAENDERUNG_ERFORDERLICH',
       'new_VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']]


# In[869]:


concat_df_buendel_linked.columns = ['BUENDEL_NUMMER', 'TypZahl', 'MK_BENENNUNG', 'SUBK', 'MSK', 'Bnd_KAT',
       'KG', 'U', 'KGU', 
       'EMISSIONSKENNZEICHEN', 'BESCHAFFUNGSART', 'SICHERHEITSRELEVANT',
       'TEILEART', 'WERKZEUGAENDERUNG_ERFORDERLICH', 
       'EE_STEUERGERAET_BETROFFEN', 'BETRIEBSANLEITUNG_BETROFFEN',
       'LEITUNGSSATZ_BETROFFEN', 'LASTENHEFTAENDERUNG_ERFORDERLICH',
       'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']


# ### Merge description with part info

# In[870]:


df_bnd_ab_abgeschlossen_dauer.columns


# In[871]:


df_buendel_linked.columns


# In[872]:


input_df = pd.merge(df_bnd_ab_abgeschlossen_dauer, concat_df_buendel_linked, left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
input_df = pd.merge(input_df, df_amount_snr,left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
input_df = pd.merge(input_df, df_amount_snr_benennung,left_on='BUENDEL_NUMMER',right_on='BUENDEL_NUMMER', how='inner')
input_df.head()


# In[873]:


input_df.columns


# In[874]:


input_df = input_df.drop(columns=['KAT','BR','AA'])


# In[875]:


input_df.columns = ['Bnd_ID', 'BUENDEL_NUMMER', 'CREATION_DATE', 'RUECKMELDE_DATUM',
                    'MODIFICATION_DATE','STATUS', 'BENENNUNG', 'STICHWORT', 'VERURSACHER',
                    'new_IST', 'new_SOLL','GEWUENSCHTER_EINSATZ', 
                    'KOSTEN_RELEVANT', 'ZERTIFIZIERUNGS_RELEVANT','PROZESS_STATUS', 
                    'VERANTWORTLICHER_FK', 'GREMIUM_ENTSCHEIDUNG_MANUELL',
                    'STATUS_GENEHMIGUNG', 'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART',
                    'BEGR_RUECKSPRUNG', 'Bnd_MK', 'Bnd_MK_BENENNUNG', 'ProduktKlasse',
                    'WERKSKENNBUCHSTABE', 'LIEFERANTEN_NUMMER', 'LIEFERANTEN_NAME',
                    'WERK_Ctime', 'ANZAHL_PAKET', 'ANZAHL_WERK', 'ANZAHL_EXT',
                    'ANZAHL_WERK_Ctime','word_count_benennung', 'word_count_stichwort', 
                    'word_count_rueck','word_count_berg_rueck', 'word_count_ist', 
                    'word_count_soll', 'BRAA','DAUER', 'DAUER_KAT', 'TypZahl', 
                    'MK_BENENNUNG', 'SUBK', 'MSK', 'Bnd_KAT', 'KG','U', 'KGU', 
                    'EMISSIONSKENNZEICHEN', 'BESCHAFFUNGSART', 'SICHERHEITSRELEVANT',
                    'TEILEART', 'WERKZEUGAENDERUNG_ERFORDERLICH', 
                    'EE_STEUERGERAET_BETROFFEN', 'BETRIEBSANLEITUNG_BETROFFEN',
                    'LEITUNGSSATZ_BETROFFEN', 'LASTENHEFTAENDERUNG_ERFORDERLICH',
                    'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN', 'ANZAHL_SACHNUMMER', 
                    'ANZAHL_ZB_BENENNUNG']


# In[876]:


input_df[['Bnd_MK','Bnd_MK_BENENNUNG','MK_BENENNUNG','KG']]


# ### Delete instances with duplicated change description
# based on 'new_IST' and 'MK' (suppose same 'new_IST' has also the same 'new_SOLL' and 'Stichwort'
# - delete duplicate data, otherwise in test data may overlap with train data
# - concat MK based on the same 'new_IST', just like above: merge MK on the same BuendelNummer

# In[877]:


input_df_final = input_df.drop_duplicates(subset=['BENENNUNG','STICHWORT','new_IST','new_SOLL'],keep='first')
input_df_final.head()


# ### Text Preprocessing + NLP Pipeline

# In[878]:


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
                    'intra','corpintra','net','itp','html','yyyy','see'])
  


# In[879]:


def plot_word_cloud(text):
    wordcloud_instance = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords=stops,
                min_font_size = 10).generate(text) 
             
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud_instance) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 


# In[880]:


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

# In[881]:


df_sbreviation_list = pd.read_excel("abbreviation_ecr_des_20200511.xlsx")
df_sbreviation_list.columns = ['Abkuerzung', 'Bedeutung']
s_sbreviation_list = set(df_sbreviation_list['Abkuerzung'].map(lambda s: s.lower()))


# In[882]:


d_all_id_sbreviations = {}
for index,row in df_sbreviation_list.iterrows():
    if row['Abkuerzung'].lower() in s_sbreviation_list:
        
        d_all_id_sbreviations[row['Abkuerzung'].lower()] = row['Bedeutung'].lower().replace(u'\xa0', u'')


# In[883]:


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


# In[884]:


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


# In[885]:


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


# In[886]:


input_df_final['cleaned_IST'] = input_df_final['new_IST_rep'].apply(lambda x: clean_text(x))
input_df_final['cleaned_SOLL'] = input_df_final['new_SOLL_rep'].apply(lambda x: clean_text(x))
input_df_final['cleaned_Stichwort'] = input_df_final['Stichwort_rep'].apply(lambda x: clean_text(x))
input_df_final['cleaned_Benennung'] = input_df_final['Benennung_rep'].apply(lambda x: clean_text(x))


# In[887]:


input_df_final.columns


# In[888]:


input_df_final_clean = input_df_final[['BUENDEL_NUMMER', 'CREATION_DATE', 'RUECKMELDE_DATUM',
                                       'MODIFICATION_DATE','STATUS', 'BENENNUNG', 'STICHWORT', 'VERURSACHER',
                                       'new_IST', 'new_SOLL','GEWUENSCHTER_EINSATZ', 'KOSTEN_RELEVANT', 
                                       'ZERTIFIZIERUNGS_RELEVANT','PROZESS_STATUS', 'VERANTWORTLICHER_FK', 
                                       'GREMIUM_ENTSCHEIDUNG_MANUELL','STATUS_GENEHMIGUNG', 
                                       'RUECKSPRUNG_KOMMENTAR', 'KOSTENBEWERTUNGS_ART','BEGR_RUECKSPRUNG', 
                                       'Bnd_MK', 'Bnd_MK_BENENNUNG', 'WERKSKENNBUCHSTABE',
                                       'LIEFERANTEN_NUMMER', 'LIEFERANTEN_NAME', 'WERK_Ctime', 'BRAA','ANZAHL_PAKET',
                                       'ANZAHL_WERK', 'ANZAHL_EXT', 'ANZAHL_WERK_Ctime', 'ANZAHL_SACHNUMMER', 
                                       'ANZAHL_ZB_BENENNUNG','DAUER',
                                       'DAUER_KAT', 'TypZahl', 'MK_BENENNUNG','MSK', 'Bnd_KAT', 'KG', 
                                       'KGU','cleaned_IST', 'cleaned_SOLL','cleaned_Stichwort', 'cleaned_Benennung',
                                       'EMISSIONSKENNZEICHEN',
                                       'BESCHAFFUNGSART', 'SICHERHEITSRELEVANT', 'TEILEART',
                                       'WERKZEUGAENDERUNG_ERFORDERLICH', 'EE_STEUERGERAET_BETROFFEN',
                                       'BETRIEBSANLEITUNG_BETROFFEN', 'LEITUNGSSATZ_BETROFFEN',
                                       'LASTENHEFTAENDERUNG_ERFORDERLICH',
                                       'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']]


# In[889]:


input_df_final_clean[['BUENDEL_NUMMER','BRAA','ANZAHL_PAKET']].head(10)


# In[890]:


input_df_final['DAUER_KAT'].nunique()


# ### Building the Machine Learning model & pipeline for Multilabel Classification
# 
# After all data exploration, let’s concentrate now on building the actual model. As it is a hierarchical multi-label classification, we need to convert our target label into a binarised vector with multiple bits set as 1.
# 
# ‘MultiLabelBinarizer’ of ‘scikit-learn’ can do that

# In[891]:


# delete duplicate strings in column ProduktArt 
for i, row in input_df_final_clean.iterrows():
    temp_typ = ';'.join(set(row['TypZahl'].split(sep=';')))
    input_df_final_clean.at[i,'TypZahl'] = temp_typ
    
# delete duplicate strings in column MK 
for i, row in input_df_final_clean.iterrows():
    temp_mk = ';'.join(set(row['MK_BENENNUNG'].split(sep=';')))
    input_df_final_clean.at[i,'MK_BENENNUNG'] = temp_mk

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
       


# In[892]:


input_df_final_clean


# In[893]:


input_df_final_clean['MK_KG'] = input_df_final_clean[['MK_BENENNUNG', 'KG']].apply(lambda x: ';'.join(x), axis=1)


# #### Wordcloud of problem description 

# In[894]:


ist_all = ''
for index, row in input_df_final_clean.iterrows():
    ist_all = ist_all + ',' + row['new_IST']
plot_word_cloud(ist_all)


# In[895]:


# with preprocessing
ist_all = ''
for index, row in input_df_final_clean.iterrows():
    ist_all = ist_all + ',' + clean_text(row['new_IST'])
plot_word_cloud(ist_all)


# ### Statistics of Data Source

# In[896]:


copy_df_kg = pd.DataFrame()
copy_df_kg['KG'] = input_df_final_clean['KG'].copy()
copy_df_kg['KG_count'] = copy_df_kg['KG'].apply(lambda x: len(str(x).split(";")))
tmp_buendel_kg_count = pd.DataFrame(copy_df_kg.groupby('KG_count').size())
tmp_buendel_kg_count


# In[897]:


copy_df_kg['KG_count'].mean()


# In[898]:


copy_df_mk = pd.DataFrame()
copy_df_mk['MK'] = input_df_final_clean['MK_BENENNUNG'].copy()
copy_df_mk['MK_count'] = copy_df_mk['MK'].apply(lambda x: len(str(x).split(";")))
tmp_buendel_mk_count = pd.DataFrame(copy_df_mk.groupby('MK_count').size())
tmp_buendel_mk_count


# In[899]:


copy_df_mk['MK_count'].mean()


# In[900]:


copy_df_msk = pd.DataFrame()
copy_df_msk['MSK'] = input_df_final_clean['MSK'].copy()
copy_df_msk['MSK_count'] = copy_df_msk['MSK'].apply(lambda x: len(str(x).split(";")))
tmp_buendel_msk_count = pd.DataFrame(copy_df_msk.groupby('MSK_count').size())
tmp_buendel_msk_count


# In[901]:


copy_df_msk['MSK_count'].mean()


# In[902]:


copy_df_mk_kg = pd.DataFrame()
copy_df_mk_kg['MK_KG'] = input_df_final_clean['MK_KG'].copy()
copy_df_mk_kg['MK_KG_count'] = copy_df_mk_kg['MK_KG'].apply(lambda x: len(str(x).split(";")))


# In[903]:


tmp_buendel_mk_kg_count = pd.DataFrame()
tmp_buendel_mk_kg_count['ECR_count']= copy_df_mk_kg.groupby('MK_KG_count').size()
tmp_buendel_mk_kg_count


# In[904]:


copy_df_mk_kg['MK_KG_count'].mean()


# In[905]:


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


plt.xlabel("Cardinality of categories (All Levels)",fontdict=font)
plt.ylabel("Number of ECRs",fontdict=font)

plt.show()


# In[906]:


df_buendel_linked.groupby(['VERANTWORTLICHER_FK'])['SACHNUMMER'].nunique().sort_values(ascending=False).head(10)


# In[907]:


df_buendel_linked.groupby('VERANTWORTLICHER_FK')['SACHNUMMER'].nunique().sort_values(ascending=False).mean()


# In[908]:


df_buendel_linked.groupby(['BUENDEL_NUMMER'])['ZB_BENENNUNG'].nunique()


# In[909]:


df_buendel_linked.groupby(['BUENDEL_NUMMER'])['SACHNUMMER'].nunique()


# In[910]:


input_df_final_clean.nunique()


# In[911]:


index_empty_zerti = input_df_final_clean.loc[input_df_final_clean['ZERTIFIZIERUNGS_RELEVANT']=='EMPTY'].index.tolist()
index_empty_zerti


# In[912]:


input_df_final_clean.drop(index_empty_zerti,inplace=True)


# ### Analyzing influencing factors

# In[913]:


input_df_final_clean.columns


# In[914]:


df_bnd_ab_abgeschlossen_dauer.columns


# In[915]:


df_bnd_ab_abgeschlossen_dauer[['BUENDEL_NUMMER','BR','AA','BRAA']]


# In[916]:


df_bnd_ab_abgeschlossen_dauer['BR'].unique()


# In[917]:


df_bnd_ab_abgeschlossen_dauer[['BUENDEL_NUMMER','ZERTIFIZIERUNGS_RELEVANT','WERK_Ctime','LIEFERANTEN_NUMMER','DAUER']].head(10)


# In[918]:


input_df_final_clean.columns


# In[919]:


input_df_final_clean[['BUENDEL_NUMMER','ZERTIFIZIERUNGS_RELEVANT','SICHERHEITSRELEVANT','ANZAHL_SACHNUMMER','DAUER']]


# In[920]:


df_bnd_ab_abgeschlossen_dauer['BUENDEL_NUMMER'].nunique()


# In[921]:


input_df_final_clean['DAUER_KAT'].unique()


# In[922]:


#29%
df_bnd_ab_abgeschlossen_dauer.loc[df_bnd_ab_abgeschlossen_dauer['DAUER']>'41 days 19:18:43.751891']['BUENDEL_NUMMER'].nunique()


# In[923]:


df_bnd_ab_abgeschlossen_dauer.loc[df_bnd_ab_abgeschlossen_dauer['DAUER']<='41 days 19:18:43.751891']['BUENDEL_NUMMER'].nunique()


# In[924]:


df_bnd_ab_abgeschlossen_dauer.loc[(df_bnd_ab_abgeschlossen_dauer['DAUER']<'41 days 19:18:43.751891')
                        & (df_bnd_ab_abgeschlossen_dauer['ANZAHL_WERK_Ctime']!=0)]['ANZAHL_WERK_Ctime'].describe()


# In[925]:


#input_df_final_clean['DAUER'].describe()


# In[926]:


input_df_final_clean.loc[input_df_final_clean['new_IST'].str.contains('Das Abschirmblech hat im hinteren Bereich eine kleine Kollision mit dem Tankspannband und eine Engstelle zur AGA')]['ANZAHL_EXT']


# ### 1. ANZAHL_SACHNUMMER

# In[927]:


input_df_final_clean['BUENDEL_NUMMER'].nunique()


# In[928]:


input_df_final_clean.loc[input_df_final_clean['DAUER']>'41 days 19:18:43.751891']['BUENDEL_NUMMER'].nunique()


# In[929]:


input_df_final_clean.loc[input_df_final_clean['DAUER']<='41 days 19:18:43.751891']['BUENDEL_NUMMER'].nunique()


# In[930]:


input_df_final_clean['DAUER'].describe()


# In[931]:


input_df_final_clean['ANZAHL_SACHNUMMER'].mean()


# In[932]:


input_df_final_clean.loc[input_df_final_clean['DAUER']>'41 days 19:18:43.751891']['ANZAHL_SACHNUMMER'].mean()


# In[933]:


input_df_final_clean.loc[input_df_final_clean['DAUER']<='41 days 19:18:43.751891']['ANZAHL_SACHNUMMER'].mean()


# In[934]:


fig = plt.figure(figsize = (35, 15))
sns.distplot(input_df_final_clean['ANZAHL_SACHNUMMER'][input_df_final_clean['DAUER_KAT']=='Longer'],color='r', label = 'longer')
sns.distplot(input_df_final_clean['ANZAHL_SACHNUMMER'][input_df_final_clean['DAUER_KAT']=='Shorter'],color='g', label = 'shorter')


# In[ ]:





# ### 2. ANZAHL_PAKET

# In[935]:


input_df_final_clean['ANZAHL_PAKET'].describe()


# In[936]:


input_df_final_clean.loc[input_df_final_clean['DAUER']>'41 days 19:18:43.751891']['ANZAHL_PAKET'].mean()


# In[937]:


input_df_final_clean.loc[input_df_final_clean['DAUER']<='41 days 19:18:43.751891']['ANZAHL_PAKET'].mean()


# In[938]:


fig = plt.figure(figsize = (35, 15))
sns.distplot(input_df_final_clean['ANZAHL_PAKET'][input_df_final_clean['DAUER_KAT']=='Longer'],color='r', label = 'longer')
sns.distplot(input_df_final_clean['ANZAHL_PAKET'][input_df_final_clean['DAUER_KAT']=='Shorter'],color='g', label = 'shorter')


# In[939]:


input_df_final_clean.loc[input_df_final_clean['BRAA']==None]


# In[940]:


#  Amount distribution of all external partners
value_count_braa = input_df_final_clean.BRAA.str.split(expand=True,pat=';').stack().value_counts()


# In[941]:


value_count_braa


# In[942]:


# Distribution of BRAA in shorter ECs
value_count_braa_shorter = input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')].BRAA.str.split(expand=True,pat=';').stack().value_counts()
value_count_braa_shorter


# In[943]:


# Distribution of BRAA in shorter ECs
value_count_braa_longer = input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')].BRAA.str.split(expand=True,pat=';').stack().value_counts()
value_count_braa_longer


# In[944]:


df_value_count_braa_shorter = value_count_braa_shorter.to_frame(name='ECR_count')
df_value_count_braa_longer = value_count_braa_longer.to_frame(name='ECR_count')


# In[945]:


df_value_count_braa = pd.concat([df_value_count_braa_shorter, df_value_count_braa_longer], axis=1)
df_value_count_braa.columns = ['Shorter','Longer']
df_value_count_braa['Shorter'] = df_value_count_braa['Shorter'].fillna(0)
df_value_count_braa['Longer'] = df_value_count_braa['Longer'].fillna(0)
df_value_count_braa


# In[946]:


from matplotlib.ticker import StrMethodFormatter
#df_value_count_ext_longer.head(20).sort_values('ECR_count').plot(kind='barh')

#ax = df_value_count_ext_longer.head(20).sort_values('ECR_count').plot(kind='barh', figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)
#df_value_count_ext_shorter.head(20).sort_values('ECR_count').plot(ax=ax, kind='barh',color='b',zorder=2,width=0.85)

#ax = df_value_count_ext.head(20).plot.barh()
ax = df_value_count_braa.sort_values('Shorter').plot(kind='barh', figsize=(8, 10), zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Amount of ECR", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Name of Affected Vehicle Projects", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# ### 3. ANZAHL_WERK

# In[947]:


input_df_final_clean['ANZAHL_WERK'] = input_df_final_clean['ANZAHL_WERK'].fillna(0)


# In[948]:


fig = plt.figure(figsize = (35, 15))
sns.distplot(input_df_final_clean['ANZAHL_WERK'][input_df_final_clean['DAUER_KAT']=='Longer'],color='r', label = 'longer')
sns.distplot(input_df_final_clean['ANZAHL_WERK'][input_df_final_clean['DAUER_KAT']=='Shorter'],color='g', label = 'shorter')


# In[949]:


#df_bnd_ab_abgeschlossen_dauer['ANZAHL_WERK'] = df_bnd_ab_abgeschlossen_dauer['ANZAHL_WERK'].fillna(0)
input_df_final_clean['ANZAHL_WERK'].mean()


# In[950]:


input_df_final_clean.loc[(input_df_final_clean['ANZAHL_WERK']!=0)]['ANZAHL_WERK'].describe()


# In[951]:


input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK']!=0)]['ANZAHL_WERK'].describe()


# In[952]:


input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK']!=0)]['ANZAHL_WERK'].describe()


# In[953]:


#  Amount distribution of all Werks
value_count_werk_shorter = input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK']!=0)].WERKSKENNBUCHSTABE.str.split(expand=True,pat=';').stack().value_counts()


# In[954]:


#  Amount distribution of all Werks
value_count_werk_longer = input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK']!=0)].WERKSKENNBUCHSTABE.str.split(expand=True,pat=';').stack().value_counts()


# In[955]:


df_value_count_werk_shorter = value_count_werk_shorter.to_frame(name='ECR_count')
df_value_count_werk_longer = value_count_werk_longer.to_frame(name='ECR_count')


# In[956]:


df_value_count_werk = pd.concat([df_value_count_werk_shorter, df_value_count_werk_longer], axis=1)
df_value_count_werk.columns = ['Shorter','Longer']
df_value_count_werk['Shorter'] = df_value_count_werk['Shorter'].fillna(0)
df_value_count_werk['Longer'] = df_value_count_werk['Longer'].fillna(0)
df_value_count_werk


# In[957]:


from matplotlib.ticker import StrMethodFormatter
#df_value_count_ext_longer.head(20).sort_values('ECR_count').plot(kind='barh')

#ax = df_value_count_ext_longer.head(20).sort_values('ECR_count').plot(kind='barh', figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)
#df_value_count_ext_shorter.head(20).sort_values('ECR_count').plot(ax=ax, kind='barh',color='b',zorder=2,width=0.85)

#ax = df_value_count_ext.head(20).plot.barh()
ax = df_value_count_werk.head(20).sort_values('Shorter').plot(kind='barh', figsize=(8, 10), zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Amount of ECR", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Name of Affected Plants", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# ### 4. ANZAHL_EXT

# In[958]:


input_df_final_clean.loc[(input_df_final_clean['ANZAHL_EXT']!=0)]['ANZAHL_EXT'].describe()


# In[959]:


input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_EXT']!=0)]['ANZAHL_EXT'].describe()


# In[960]:


input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_EXT']!=0)]['ANZAHL_EXT'].describe()


# In[961]:


#  Amount distribution of all external partners
value_count_ext_shorter = input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_EXT']!=0)].LIEFERANTEN_NAME.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[962]:


#  Amount distribution of all external partners
value_count_ext_nummer_shorter = input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_EXT']!=0)].LIEFERANTEN_NUMMER.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[963]:


value_count_ext_longer = input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_EXT']!=0)].LIEFERANTEN_NAME.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[964]:


#  Amount distribution of all external partners
value_count_ext_nummer_longer = input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_EXT']!=0)].LIEFERANTEN_NUMMER.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[965]:


df_value_count_ext_longer = value_count_ext_longer.to_frame(name='ECR_count')


# In[966]:


df_value_count_ext_nummer_longer = value_count_ext_nummer_longer.to_frame(name='ECR_count')


# In[967]:


df_value_count_ext_shorter = value_count_ext_shorter.to_frame(name='ECR_count')


# In[968]:


df_value_count_ext_nummer_shorter = value_count_ext_nummer_shorter.to_frame(name='ECR_count')


# In[969]:


df_value_count_ext_nummer_shorter


# In[970]:


df_value_count_ext_shorter


# In[971]:


df_value_count_ext_longer


# In[972]:


df_value_count_ext_nummer_longer


# In[973]:


df_value_count_ext = pd.concat([df_value_count_ext_shorter, df_value_count_ext_longer], axis=1)
df_value_count_ext.columns = ['Shorter','Longer']
df_value_count_ext['Shorter'] = df_value_count_ext['Shorter'].fillna(0)
df_value_count_ext['Longer'] = df_value_count_ext['Longer'].fillna(0)
df_value_count_ext


# In[974]:


df_value_count_ext_nummer = pd.concat([df_value_count_ext_nummer_shorter, df_value_count_ext_nummer_longer], axis=1)
df_value_count_ext_nummer.columns = ['Shorter','Longer']
df_value_count_ext_nummer['Shorter'] = df_value_count_ext_nummer['Shorter'].fillna(0)
df_value_count_ext_nummer['Longer'] = df_value_count_ext_nummer['Longer'].fillna(0)
df_value_count_ext_nummer


# In[975]:


from matplotlib.ticker import StrMethodFormatter
#df_value_count_ext_longer.head(20).sort_values('ECR_count').plot(kind='barh')

#ax = df_value_count_ext_longer.head(20).sort_values('ECR_count').plot(kind='barh', figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)
#df_value_count_ext_shorter.head(20).sort_values('ECR_count').plot(ax=ax, kind='barh',color='b',zorder=2,width=0.85)

#ax = df_value_count_ext.head(20).plot.barh()
ax = df_value_count_ext_nummer.head(20).sort_values('Shorter').plot(kind='barh', figsize=(8, 10), zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Amount of ECR", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Name of External Partners", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# In[976]:


input_df_final_clean['DAUER']


# In[977]:


input_df_final_clean.dtypes


# In[978]:


input_df_final_clean.loc[(input_df_final_clean['ANZAHL_EXT']!=0)].LIEFERANTEN_NUMMER.str.split(expand=True,pat=';').stack().value_counts().drop(index='None').head(10)


# In[979]:


# Calculate average processing time for each external partners
tmp_df_x = input_df_final_clean.copy()
tmp_df_x['DAUER_Int'] = tmp_df_x['DAUER'].values
#tmp_df_x.groupby('LIEFERANTEN_NUMMER')['DAUER']


# In[980]:


tmp_df_x['WERKSKENNBUCHSTABE']


# In[981]:


tmp_df_x.loc[tmp_df_x['LIEFERANTEN_NAME']=='Autoneum Mexico Operations S.A de C.V.']['DAUER_Int'].mean()


# In[1515]:


tmp_df_x.loc[tmp_df_x['WERKSKENNBUCHSTABE']=='FL']['DAUER_Int'].describe()


# In[1497]:


tmp_df_x.loc[tmp_df_x['LIEFERANTEN_NUMMER']=='10940211']['BUENDEL_NUMMER'].nunique()


# ### 5. ANZAHL_WERK_Ctime

# In[983]:


input_df_final_clean.loc[(input_df_final_clean['ANZAHL_WERK_Ctime']!=0)]['ANZAHL_WERK_Ctime'].describe()


# In[984]:


input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK_Ctime']!=0)]['ANZAHL_WERK_Ctime'].describe()


# In[985]:


input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK_Ctime']!=0)]['ANZAHL_WERK_Ctime'].describe()


# In[986]:


#  Amount distribution of all external partners
value_count_werkctime = input_df_final_clean.loc[input_df_final_clean['ANZAHL_WERK_Ctime']!=0].WERK_Ctime.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[987]:


value_count_werkctime_shorter = input_df_final_clean.loc[(input_df_final_clean['DAUER']<='41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK_Ctime']!=0)].WERK_Ctime.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[988]:


value_count_werkctime_longer = input_df_final_clean.loc[(input_df_final_clean['DAUER']>'41 days 19:18:43.751891')
                        & (input_df_final_clean['ANZAHL_WERK_Ctime']!=0)].WERK_Ctime.str.split(expand=True,pat=';').stack().value_counts().drop(index='None')


# In[989]:


df_value_count_werkctime_longer = value_count_werkctime_longer.to_frame(name='ECR_count')
df_value_count_werkctime_longer


# In[990]:


df_value_count_werkctime_shorter = value_count_werkctime_shorter.to_frame(name='ECR_count')
df_value_count_werkctime_shorter


# In[991]:


df_value_count_werctime = pd.concat([df_value_count_werkctime_shorter, df_value_count_werkctime_longer], axis=1)
df_value_count_werctime.columns = ['Shorter','Longer']
df_value_count_werctime['Shorter'] = df_value_count_werctime['Shorter'].fillna(0)
df_value_count_werctime['Longer'] = df_value_count_werctime['Longer'].fillna(0)
df_value_count_werctime


# In[ ]:





# In[992]:


from matplotlib.ticker import StrMethodFormatter

ax = df_value_count_werctime.head(20).sort_values('Shorter').plot(kind='barh', figsize=(8, 10), zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Amount of ECR", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Name of Affected Plants", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# ### 6. Teileart

# In[993]:


input_df_final_clean.groupby(['TEILEART'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[994]:


input_df_final_clean.loc[input_df_final_clean['DAUER']<='41 days 19:18:43.751891'].groupby(
    ['TEILEART'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[995]:


input_df_final_clean.loc[input_df_final_clean['DAUER']>'41 days 19:18:43.751891'].groupby(
    ['TEILEART'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[996]:


g = sns.catplot(x="TEILEART",col ='DAUER_KAT',kind="count", data=input_df_final_clean,height=5,aspect=1.5)
g.set_xticklabels(rotation=45)


# In[997]:


x,y = 'TEILEART', 'DAUER_KAT'
(input_df_final_clean
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar',height=5,aspect=2)).set_xticklabels(rotation=45)


# ### 7. Bnd_MK

# In[998]:


tmp_df_x_it = input_df_final_clean.copy()
tmp_df_x_it.columns


# In[999]:


def dauer_cat_tmp(row):
    if row['DAUER'] <= pd.to_timedelta('41 days'):
        val = 'Shorter'
    #elif (row['DAUER'] > pd.to_timedelta('42 days')) & (row['DAUER'] <= pd.to_timedelta('64 days')):
        #val = 'Q2'
#     elif (row['DAUER'] > pd.to_timedelta('33 days')) & (row['DAUER'] <= pd.to_timedelta('64 days')):
#         val = 'Q3'
    else:
        val = 'Longer'
    return val

tmp_df_x_it['DAUER_KAT_tmp'] = tmp_df_x_it.apply(dauer_cat_tmp, axis=1)


# In[1000]:


tmp_df_x_it[['DAUER','DAUER_KAT_tmp']]


# In[1001]:


x,y = 'Bnd_MK', 'DAUER_KAT_tmp'
(tmp_df_x_it
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar',height=5,aspect=5)).set_xticklabels(rotation=90)


# In[1002]:


x,y = 'VERURSACHER', 'DAUER_KAT_tmp'
(tmp_df_x_it
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar',height=5,aspect=5)).set_xticklabels(rotation=90)


# In[1003]:


x,y = 'Bnd_MK_BENENNUNG', 'DAUER_KAT'
(input_df_final_clean
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar',height=5,aspect=5)).set_xticklabels(rotation=90)


# In[1004]:


input_df_final_clean.groupby(['Bnd_MK_BENENNUNG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False).head(20)


# In[1005]:


input_df_final_clean.loc[input_df_final_clean['DAUER']>'41 days 19:18:43.751891'].groupby(
    ['Bnd_MK_BENENNUNG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False).head(20)


# In[1006]:


input_df_final_clean.loc[input_df_final_clean['DAUER']<='41 days 19:18:43.751891'].groupby(
    ['Bnd_MK_BENENNUNG'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# ### 8. Sicherheit Relevant

# In[397]:


df_bnd_ab_abgeschlossen_dauer.columns


# In[398]:


# JA: 29% 
input_df_final_clean.groupby(['SICHERHEITSRELEVANT'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[399]:


input_df_final_clean.loc[input_df_final_clean['SICHERHEITSRELEVANT']=='JA']['DAUER'].describe()


# In[400]:


input_df_final_clean.loc[input_df_final_clean['SICHERHEITSRELEVANT']=='NEIN']['DAUER'].describe()


# In[401]:


# JA: 30%
input_df_final_clean.loc[input_df_final_clean['DAUER']<='59 days 19:18:43.751891'].groupby(
    ['SICHERHEITSRELEVANT'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[402]:


# JA: 26%
input_df_final_clean.loc[input_df_final_clean['DAUER']>'59 days 19:18:43.751891'].groupby(
    ['SICHERHEITSRELEVANT'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# ### 9. ZERTIFIZIERUNGS_RELEVANT

# In[403]:


input_df_final_clean.groupby(['ZERTIFIZIERUNGS_RELEVANT'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[404]:


input_df_final_clean.loc[input_df_final_clean['ZERTIFIZIERUNGS_RELEVANT']=='JA']['DAUER'].describe()


# In[405]:


input_df_final_clean.loc[input_df_final_clean['ZERTIFIZIERUNGS_RELEVANT']=='NEIN']['DAUER'].describe()


# In[406]:


input_df_final_clean.loc[input_df_final_clean['DAUER']<='59 days 19:18:43.751891'].groupby(
    ['ZERTIFIZIERUNGS_RELEVANT'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[407]:


input_df_final_clean.loc[input_df_final_clean['DAUER']>'59 days 19:18:43.751891'].groupby(
    ['ZERTIFIZIERUNGS_RELEVANT'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# ## Predict influecing factors of long lead time of ECR

# In[1007]:


df_bnd_ab_abgeschlossen_dauer['DAUER_KAT'].describe()


# In[1008]:


input_df_final['DAUER_KAT'].describe()


# In[1009]:


input_df_final_clean['LIEFERANTEN_NUMMER'] = input_df_final_clean['LIEFERANTEN_NUMMER'].fillna('None')
input_df_final_clean['WERK_Ctime'] = input_df_final_clean['WERK_Ctime'].fillna('None')
input_df_final_clean['WERKSKENNBUCHSTABE'] = input_df_final_clean['WERKSKENNBUCHSTABE'].fillna('None')

input_df_final_clean['LIEFERANTEN_NUMMER'] = input_df_final_clean['LIEFERANTEN_NUMMER'].astype('str')
input_df_final_clean['WERK_Ctime'] = input_df_final_clean['WERK_Ctime'].astype('str')
input_df_final_clean['WERKSKENNBUCHSTABE'] = input_df_final_clean['WERKSKENNBUCHSTABE'].astype('str')


# In[413]:


# # Encoding multi-label features: external partner numbers, Werk_Ctime, Werk
# from sklearn.preprocessing import MultiLabelBinarizer

# def encode_multilabel_features(s):
#     ext_nummers = []
#     ext_nummers = [
#             row['LIEFERANTEN_NUMMER'].split(";")
#             for index,row in input_df_final_clean.iterrows()
#             ]

#     werk_ctime = []
#     werk_ctime = [
#             row['WERK_Ctime'].split(";")
#             for index,row in input_df_final_clean.iterrows()
#             ]


#     werk = []
#     werk = [
#             row['WERKSKENNBUCHSTABE'].split(";")
#             for index,row in input_df_final_clean.iterrows()
#             ]
    
#     for i in ext_nummers:
#         if 'None' in i:
#             i.remove('None')
#         if 'nan' in i:
#             i.remove('nan')

#     for j in werk_ctime:
#         if 'None' in j:
#             j.remove('None')
#         if 'nan' in j:
#             j.remove('nan')

#     for k in werk:
#         if 'None' in k:
#             k.remove('None')
#         if 'nan' in k:
#             k.remove('nan')
    
#     encode_ext_nummers = []    
#     mlb_ext = MultiLabelBinarizer()
#     encode_ext_nummers = mlb_ext.fit_transform(ext_nummers)
    
#     encode_werkctime = []    
#     mlb_werkctime = MultiLabelBinarizer()
#     encode_werkctime = mlb_werkctime.fit_transform(werk_ctime)
    
#     encode_werk = []    
#     mlb_werk = MultiLabelBinarizer()
#     encode_werk = mlb_werk.fit_transform(werk)

#     encoded_multi_feature = np.hstack((encode_ext_nummers,encode_werkctime,encode_werk))
#     multi_feature_names = np.hstack((mlb_ext.classes_,mlb_werkctime.classes_,mlb_werk.classes_))
#     return encoded_multi_feature, multi_feature_names


# In[414]:


#encoded_multi_feature, multi_feature_names = encode_multilabel_features(input_df_final_clean)


# In[415]:


#encoded_multi_feature


# In[416]:


#multi_feature_names


# In[1010]:


# encoding multi-label features: Ext, Werk, WerkCtime into numeric values, and append to the input dataframe as features
# these features can be selected as numeric features in numeric pipeline

df_encoded_ext = input_df_final_clean['LIEFERANTEN_NUMMER'].str.get_dummies(';')
df_encoded_ext = df_encoded_ext.add_prefix('EXT_')

df_encoded_werk_ctime= input_df_final_clean['WERK_Ctime'].str.get_dummies(';')
df_encoded_werk_ctime = df_encoded_werk_ctime.add_prefix('Werk_Ctime_')

df_encoded_werk = input_df_final_clean['WERKSKENNBUCHSTABE'].str.get_dummies(';')
df_encoded_werk = df_encoded_werk.add_prefix('WKB_')


# In[1011]:


df_encoded_multi_feature = pd.concat([df_encoded_ext,df_encoded_werk_ctime,df_encoded_werk], axis=1)


# In[1012]:


df_encoded_multi_feature


# In[645]:


# get column names of data frame in a list
col_names = list(df_encoded_multi_feature)
print("\nNames of dataframe columns")
print(col_names)


# In[1013]:


# loop to change each column to category type
for col in col_names:
    df_encoded_multi_feature[col] = df_encoded_multi_feature[col].astype('category',copy=False)

print("\nExample data changed to category type")


# In[1014]:


df_encoded_multi_feature.head(5)


# In[1015]:


input_df_final_clean[input_df_final_clean['ANZAHL_EXT'] < 0] = 0
input_df_final_clean[input_df_final_clean['ANZAHL_WERK'] < 0] = 0
input_df_final_clean[input_df_final_clean['ANZAHL_WERK_Ctime'] < 0] = 0

input_df_final_clean['ANZAHL_EXT'] = input_df_final_clean['ANZAHL_EXT'].fillna(0)
input_df_final_clean['ANZAHL_WERK'] = input_df_final_clean['ANZAHL_WERK'].fillna(0)
input_df_final_clean['ANZAHL_WERK_Ctime'] = input_df_final_clean['ANZAHL_WERK_Ctime'].fillna(0)

input_df_final_clean['ANZAHL_EXT'] = input_df_final_clean['ANZAHL_EXT'].astype('int')
input_df_final_clean['ANZAHL_WERK'] = input_df_final_clean['ANZAHL_WERK'].astype('int')
input_df_final_clean['ANZAHL_WERK_Ctime'] = input_df_final_clean['ANZAHL_WERK_Ctime'].astype('int')


# In[1016]:


input_df_final_clean_new = pd.concat([input_df_final_clean,df_encoded_multi_feature], axis=1)


# In[1017]:


input_df_final_clean_new = input_df_final_clean_new.fillna(0)


# In[1018]:


rows_with_nan = []
for index, row in input_df_final_clean_new.iterrows():
    is_nan_series = row.isnull()
    if is_nan_series.any():
        rows_with_nan.append(index)

print(rows_with_nan)


# In[1019]:


input_df_final_clean_new.dtypes


# In[1020]:


input_df_final_clean['ANZAHL_WERK_Ctime'].describe()


# In[1021]:


input_df_final_clean.loc[input_df_final_clean['LIEFERANTEN_NUMMER']=='None']['ANZAHL_EXT']


# In[1022]:


input_df_final_clean.loc[input_df_final_clean['WERK_Ctime']=='None']['ANZAHL_WERK_Ctime']


# In[1023]:


input_df_final_clean['DAUER_KAT'].nunique()


# In[1024]:


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
    
    


# In[1025]:


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


# In[1026]:


import category_encoders as ce
#hc_features = ['VERANTWORTLICHER_FK', 'VERURSACHER', 'Bnd_MK_BENENNUNG', 'GREMIUM_ENTSCHEIDUNG_MANUELL']
hc_features = ['VERURSACHER', 'Bnd_MK_BENENNUNG']
hc_pipeline = make_pipeline(ce.TargetEncoder())
# hc_pipeline_verantwortlicher = make_pipeline(ce.HashingEncoder(n_components=1000))
# hc_pipeline_verursacher = make_pipeline(ce.HashingEncoder(n_components=30))
# hc_pipeline_mk = make_pipeline(ce.HashingEncoder(n_components=100))
# hc_pipeline_gre = make_pipeline(ce.HashingEncoder(n_components=40))


print(f'N hc_features: {len(hc_features)} \n')
print(', '.join(hc_features))


# In[1027]:


# One-Hot Encoding: Modulecomponent of Bündle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer

# #oh_features = ['Bnd_MK_BENENNUNG']
# oh_features = []
# oh_features = ['KOSTEN_RELEVANT', 'ZERTIFIZIERUNGS_RELEVANT', 'EMISSIONSKENNZEICHEN', 
#                'BESCHAFFUNGSART', 'SICHERHEITSRELEVANT', 'TEILEART', 'EE_STEUERGERAET_BETROFFEN', 
#                'BETRIEBSANLEITUNG_BETROFFEN', 'LEITUNGSSATZ_BETROFFEN', 'LASTENHEFTAENDERUNG_ERFORDERLICH', 
#                'VERSUCHSTEIL_ASACHNUMMER_BETROFFEN']
#oh_pipeline = make_pipeline(SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore'))

MAX_OH_CARDINALITY = 200

def select_oh_features(df):
    
    oh_features =        df        .select_dtypes(['object', 'category'])        .apply(lambda col: col.nunique())        .loc[lambda x: x < MAX_OH_CARDINALITY ]        .index        .tolist()
        
    return oh_features

oh_features = []
oh_features = select_oh_features(input_df_final_clean)
#oh_features = ['VERURSACHER', 'Bnd_MK_BENENNUNG']
oh_features.remove('STATUS')
oh_features.remove('KOSTEN_RELEVANT')
oh_features.remove('ZERTIFIZIERUNGS_RELEVANT')
oh_features.remove('PROZESS_STATUS')
oh_features.remove('STATUS_GENEHMIGUNG')
oh_features.remove('DAUER_KAT')
oh_features.remove('EMISSIONSKENNZEICHEN')
oh_features.remove('BESCHAFFUNGSART')
oh_features.remove('SICHERHEITSRELEVANT')
oh_features.remove('TEILEART')
oh_features.remove('KOSTENBEWERTUNGS_ART')

oh_features.remove('WERKZEUGAENDERUNG_ERFORDERLICH')
oh_features.remove('EE_STEUERGERAET_BETROFFEN')
oh_features.remove('BETRIEBSANLEITUNG_BETROFFEN')
oh_features.remove('LEITUNGSSATZ_BETROFFEN')
oh_features.remove('LASTENHEFTAENDERUNG_ERFORDERLICH')
oh_features.remove('VERSUCHSTEIL_ASACHNUMMER_BETROFFEN')

oh_features.remove('GEWUENSCHTER_EINSATZ')
oh_features.remove('GREMIUM_ENTSCHEIDUNG_MANUELL')
oh_features.remove('Bnd_MK')
oh_features.remove('Bnd_KAT')
oh_features.remove('VERURSACHER')


oh_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

print(f'N oh_features: {len(oh_features)} \n')
print(', '.join(oh_features))


# In[1252]:


# Numeric features
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer

numeric_features = []

#numeric_features = select_numeric_features(input_df_final_clean_new)
#numeric_features.remove('DAUER')
#numeric_features.remove('ANZAHL_ZB_BENENNUNG')
# numeric_features.remove('EXT_None')
# numeric_features.remove('EXT_nan')
# numeric_features.remove('Werk_Ctime_None')
# numeric_features.remove('Werk_Ctime_nan')
# numeric_features.remove('WKB_nan')


numeric_features = ['ANZAHL_PAKET', 'ANZAHL_WERK_Ctime', 'ANZAHL_WERK','ANZAHL_EXT', 'ANZAHL_SACHNUMMER']
#numeric_features = ['ANZAHL_PAKET', 'ANZAHL_SACHNUMMER']
numeric_pipeline = SelectColumns(numeric_features)

print(f'N numeric_features: {len(numeric_features)} \n')
print(', '.join(numeric_features))


# In[1253]:


from sklearn.compose import ColumnTransformer
column_transformer =    ColumnTransformer(transformers=                          [('txt_ist_pipeline', TfidfVectorizer(min_df=2,stop_words=stops), 'cleaned_IST'),                           ('txt_benennung_pipeline', TfidfVectorizer(min_df=3,stop_words=stops), 'cleaned_Benennung'),                           ('txt_stichwort_pipeline', TfidfVectorizer (min_df=3, stop_words=stops), 'cleaned_Stichwort'),                          #('oh_pipeline', OneHotEncoder(handle_unknown='ignore'), oh_features),\
                          ('hc_pipeline', ce.TargetEncoder(), hc_features),\
                           ('numeric_pipeline', numeric_pipeline, numeric_features)],\
                            n_jobs=-1,remainder='drop')


# ### Predictin processing time of new Bündel

# In[1254]:


input_df_final_clean.groupby(['DAUER_KAT']).nunique()


# In[1222]:


df_x_it = pd.DataFrame(input_df_final_clean.copy())


# In[ ]:





# In[925]:


# filter_col = [col for col in df_x_it if (col.startswith('EXT_')) | (col.startswith('Werk_Ctime_')) | (col.startswith('WKB_'))]
# filter_col


# In[926]:


#df_x_it[filter_col]=df_x_it[filter_col].astype('category')


# In[1223]:


input_df_final_clean_new['ANZAHL_WERK_Ctime'].describe()


# In[1224]:


df_x_it.dtypes


# In[1225]:


df_x_it['cleaned_IST'] = df_x_it['cleaned_IST'].astype('str')
df_x_it['cleaned_Benennung'] = df_x_it['cleaned_Benennung'].astype('str')
df_x_it['cleaned_Stichwort'] = df_x_it['cleaned_Stichwort'].astype('str')

#df_x_it['LIEFERANTEN_NUMMER'] = df_x_it['LIEFERANTEN_NUMMER'].astype('str')
#df_x_it['WERK_Ctime'] = df_x_it['WERK_Ctime'].astype('str')
#df_x_it['WERKSKENNBUCHSTABE'] = df_x_it['WERKSKENNBUCHSTABE'].astype('str')

df_x_it['VERURSACHER'] = df_x_it['VERURSACHER'].fillna('None')
df_x_it['VERURSACHER'] = df_x_it['VERURSACHER'].astype('category')
df_x_it['Bnd_MK_BENENNUNG'] = df_x_it['Bnd_MK_BENENNUNG'].astype('category')

# df_x_it['KOSTEN_RELEVANT'] = df_x_it['KOSTEN_RELEVANT'].astype('category')
# df_x_it['ZERTIFIZIERUNGS_RELEVANT'] = df_x_it['ZERTIFIZIERUNGS_RELEVANT'].astype('category')
# df_x_it['GREMIUM_ENTSCHEIDUNG_MANUELL'] = df_x_it['GREMIUM_ENTSCHEIDUNG_MANUELL'].astype('category')
#df_x_it['Bnd_MK_BENENNUNG'] = df_x_it['Bnd_MK_BENENNUNG'].fillna('None')
#df_x_it['Bnd_MK_BENENNUNG'] = df_x_it['Bnd_MK_BENENNUNG'].astype('category')
# df_x_it['EMISSIONSKENNZEICHEN'] = df_x_it['EMISSIONSKENNZEICHEN'].astype('category')
# df_x_it['BESCHAFFUNGSART'] = df_x_it['BESCHAFFUNGSART'].astype('category')
# df_x_it['SICHERHEITSRELEVANT'] = df_x_it['SICHERHEITSRELEVANT'].astype('category')
# df_x_it['TEILEART'] = df_x_it['TEILEART'].astype('category')
# df_x_it['WERKZEUGAENDERUNG_ERFORDERLICH'] = df_x_it['WERKZEUGAENDERUNG_ERFORDERLICH'].astype('category')
# df_x_it['EE_STEUERGERAET_BETROFFEN'] = df_x_it['EE_STEUERGERAET_BETROFFEN'].astype('category')
# df_x_it['BETRIEBSANLEITUNG_BETROFFEN'] = df_x_it['BETRIEBSANLEITUNG_BETROFFEN'].astype('category')
# df_x_it['LEITUNGSSATZ_BETROFFEN'] = df_x_it['LEITUNGSSATZ_BETROFFEN'].astype('category')
# df_x_it['LASTENHEFTAENDERUNG_ERFORDERLICH'] = df_x_it['LASTENHEFTAENDERUNG_ERFORDERLICH'].astype('category')
# df_x_it['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'] = df_x_it['VERSUCHSTEIL_ASACHNUMMER_BETROFFEN'].astype('category')
df_x_it['DAUER_KAT_num'] = df_x_it['DAUER_KAT'].astype('category').cat.codes


# In[1226]:


df_x_it['Bnd_MK_BENENNUNG']


# In[1227]:


y_it = df_x_it['DAUER_KAT']


# In[1228]:


df_x_it.loc[df_x_it['VERURSACHER'].isna()]


# In[1229]:


from sklearn.model_selection import train_test_split
train_x_it, test_x_it, train_y_it, test_y_it = train_test_split(df_x_it, y_it,test_size=0.2, random_state=42)


# In[1402]:


df_x_it.to_csv("dlz_df_x_20210114.csv",encoding='utf-8-sig')


# In[1403]:


train_x_it.to_csv("dlz_train_x_20210114.csv",encoding='utf-8-sig')


# In[1480]:


train_y_it.to_csv("dlz_train_y_20210114.csv",encoding='utf-8-sig')


# In[1255]:


df_x_it_pre = column_transformer.fit_transform(df_x_it,df_x_it['DAUER_KAT_num'])
#df_x_it_pre = column_transformer.fit_transform(df_x_it)
df_x_it_pre.shape


# In[1256]:


train_it_pre = column_transformer.transform(train_x_it)


# In[1482]:


train_x_it.shape


# In[1473]:


#train_it_pre.to_csv("dlz_train_x_pre_20210114.csv",encoding='utf-8-sig')


# In[1257]:


test_it_pre = column_transformer.transform(test_x_it)


# In[1477]:


train_x_it.shape


# In[1478]:


test_it_pre.shape


# In[1488]:


type(train_y_it)


# In[1260]:


train_it_pre.toarray()[-3:]


# In[1261]:


df_train_pre = pd.DataFrame(train_it_pre.toarray())
df_train_pre.head()


# In[1262]:


train_x_it[['VERURSACHER', 'Bnd_MK','Bnd_MK_BENENNUNG','DAUER_KAT']].head()


# In[1263]:


df_x_it.columns[0:25]


# In[1264]:


input_df_final_clean_new[['DAUER','DAUER_KAT']]


# In[1265]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier

import time
base_rf = RandomForestClassifier(bootstrap=True, criterion='gini', 
                                 random_state=42, n_estimators=200, n_jobs=-1, verbose=True)

svc = svm.SVC()
lin_svc = svm.LinearSVC()
lg = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=3)

parameters = [
    {
        'criterion': ['gini', 'entropy'],
          
    }
]

# #score = 'f1_micro'
score = 'accuracy'
clf = GridSearchCV(base_rf, parameters, scoring=score,cv=3, verbose=True)
clf.fit(train_it_pre, train_y_it)

#lin_svc.fit(train_it_pre, train_y_it)
#lg.fit(train_it_pre, train_y_it)
#knn.fit(train_it_pre, train_y_it)
start=time.time()


print('training time taken: ',round(time.time()-start,0),'seconds')


# In[1406]:


import joblib
joblib.dump(clf, 'dlz_ab_20210114.pkl')


# In[1266]:


clf.best_params_


# In[1267]:


clf.classes_


# In[1268]:


clf.cv_results_


# In[1269]:


means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
print()


# In[1270]:


from sklearn.metrics import classification_report

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = test_y_it, clf.predict(test_it_pre)
print(classification_report(y_true, y_pred))
print()


# ### Visualize Random Forest

# In[465]:


from sklearn.tree import export_graphviz


# In[466]:


len(clf.best_estimator_.estimators_)


# In[ ]:





# In[467]:


from sklearn.model_selection import cross_val_score
cross_val_score(clf,train_it_pre, train_y_it)


# In[1271]:


pred_it = clf.predict(test_it_pre)


# In[1272]:


test_y_it[0:10]


# In[1273]:


pred_it[0:10]


# In[1274]:


pred_it_proba = clf.predict_proba(test_it_pre)


# In[1275]:


pred_it_proba


# In[1276]:


print("Accuracy = ",round(accuracy_score(test_y_it,pred_it),ndigits=3))
#print("Accuracy_per Class = ",round(accuracy_score(test_y_it,pred_it),ndigits=3))

print("Micro_Precision = ",round(precision_score(test_y_it, pred_it, average='micro'),ndigits=3))
print("Precision_per class = ",precision_score(test_y_it, pred_it, average=None))
print('\n')

print("Micro_Recall = ",round(recall_score(test_y_it, pred_it, average='micro'),ndigits=3))
print("Weighted_Recall = ",round(recall_score(test_y_it, pred_it, average='weighted'),ndigits=3))
print('\n')


# In[1277]:


# Create a dictionary with most important features
importances = clf.best_estimator_.feature_importances_
importances


# In[1278]:


importances.shape


# In[1279]:


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


# In[1280]:


feature_names = np.array(get_feature_names(column_transformer))
for i,element in enumerate(feature_names):
    feature_names[i] = feature_names[i].replace('pipeline__','') 
    feature_names[i] = feature_names[i].replace('x0','MK')
#     feature_names[i] = feature_names[i].replace('x1','MK')
#     feature_names[i] = feature_names[i].replace('x2','EMISSIONKENNENZEICHEN')
#     feature_names[i] = feature_names[i].replace('x3','BESCHAFFUNGSART')
#     feature_names[i] = feature_names[i].replace('x4','SICHERHEITSRELEVANT')
#     feature_names[i] = feature_names[i].replace('x5','TEILEART')
#     #feature_names[i] = feature_names[i].replace('x6','WERKZEUGAENDERUNG_ERFORDERLICH')
#     feature_names[i] = feature_names[i].replace('x6','EE_STEUERGERAET_BETROFFEN')
#     feature_names[i] = feature_names[i].replace('x7','BETRIEBSANLEITUNG_BETROFFEN')
#     feature_names[i] = feature_names[i].replace('x8','LEITUNGSSATZ_BETROFFEN')
#     feature_names[i] = feature_names[i].replace('x9','LASTENHEFTAENDERUNG_ERFORDERLICH')
#     feature_names[i] = feature_names[i].replace('x10','VERSUCHSTEIL_ASACHNUMMER_BETROFFEN')
    
    


# In[1281]:


feature_names


# In[958]:


import os
import pydot
from sklearn.tree import export_graphviz

class_names = ['Longer','Shorter']

export_graphviz(clf.best_estimator_.estimators_[1], out_file = 'tree.dot',
               feature_names=feature_names,
               class_names=class_names,
               rounded=True, proportion=False,
               precision=2, filled=True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
# from IPython.display import Image
# Image(filename = 'tree.png')


# In[ ]:





# In[1288]:


top_indices = np.argsort(importances)[::-1][:7]
top_indices


# In[1289]:


words_importance = pd.DataFrame({'most_important_words': feature_names[top_indices],
                                 'importance': importances[top_indices]})
words_importance


# In[1290]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.xkcd(length=100)

# Create a dictionary with most important features
# importances = pipeline.named_steps['randomforestclassifier'].feature_importances_
# top_indices = np.argsort(importances)[::-1][:10]
# features_names = np.array(vectorizer.get_feature_names())
# words_importance = pd.DataFrame({'most_important_words': features_names[top_indices],
#                                  'importance': importances[top_indices]})
plt.figure(figsize=(15,10)) 
plt.bar(x=feature_names[top_indices], height=importances[top_indices])
plt.xlabel("most_important_features")
plt.ylabel("importance")
plt.xticks(rotation=90)
plt.title('Global features importance')


# In[1291]:


feature_names


# In[1292]:


clf.classes_


# In[ ]:


#del explainer


# In[1293]:


from collections import OrderedDict
#from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

class_names = ['Longer','Shorter']
#explainer = LimeTextExplainer(class_names=class_names)
explainer = LimeTabularExplainer(train_it_pre,feature_names=feature_names, 
                                 class_names=class_names,discretize_continuous=True, random_state=2)
#explanation = explainer.explain_instance(train_it_pre, clf.predict_proba, num_features=6)

# explanation.as_pyplot_figure();
# plt.show()


# In[1198]:


input_df_final_clean.loc[input_df_final_clean['new_IST'].str.contains('Aktuelle NVH')]


# In[1199]:


train_x_it.loc[train_x_it['new_IST'].str.contains('Das Abschirmblech hat im hinteren')]['ANZAHL_SACHNUMMER']


# In[ ]:





# In[1200]:


train_it_pre.toarray()


# In[1434]:


test_x_it.head(10)[['BUENDEL_NUMMER','ANZAHL_SACHNUMMER']]


# In[1442]:



#test_sample = test_x_it.loc[test_x_it['new_IST'].str.contains('Aktuelle NVH')]
#test_sample = train_x_it.loc[train_x_it['new_IST'].str.contains('Das Abschirmblech hat im hinteren')]
test_sample = test_x_it.loc[test_x_it['BUENDEL_NUMMER']=='0070425-004']
test_sample_pre = column_transformer.transform(test_sample)
#exp = explainer.explain_instance(test_it_pre[idx], clf.predict_proba, num_features=20)


# In[1443]:


exp = explainer.explain_instance(test_sample_pre, clf.predict_proba, num_features=8)


# In[1438]:


pd.set_option('display.max_colwidth', -1)


# In[1439]:


test_sample.columns


# In[1459]:


df_buendel_linked.columns


# In[1472]:


df_buendel_linked.loc[df_buendel_linked['BUENDEL_NUMMER']=='0070425-004'][['LIEFERANTEN_NUMMER','LIEFERANTEN_NAME']]


# In[1456]:


test_sample['Bnd_MK_BENENNUNG']


# In[1159]:


test_sample_pre


# In[1298]:


exp.as_list()


# In[1445]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp.as_pyplot_figure()


# In[1300]:


exp.class_names


# In[1446]:


exp.show_in_notebook(labels=None)


# In[1401]:


exp.save_to_file('exp.html')


# ### Feature Correlation

# In[1302]:


df_x_it_pre


# In[1303]:


df_input_pre = pd.DataFrame(df_x_it_pre.toarray(), columns=list(feature_names))
df_input_pre


# In[1304]:


df_x_it.groupby(['DAUER_KAT_num'])['BUENDEL_NUMMER'].nunique().sort_values(ascending=False)


# In[1305]:


df_input_pre['DAUER_KAT_num']=df_x_it['DAUER_KAT_num']


# In[1306]:


col_names = df_input_pre.columns.tolist()
col_names


# In[1307]:


list_non_txt_cols =[]
len_cols = len(col_names)

for i in col_names:
    if i.startswith('hc_'):
        list_non_txt_cols.append(i)
    
    if i.startswith('oh_'):
        list_non_txt_cols.append(i)
        
    if i.startswith('numeric_'):
        list_non_txt_cols.append(i)


# In[1308]:


list_non_txt_cols


# In[1309]:


list_non_txt_cols.append('DAUER_KAT_num')


# In[1490]:


df_buendel_linked.columns


# In[1310]:


corr_non_text = df_input_pre[list_non_txt_cols].corr()


# In[1311]:


corr_non_text


# In[1191]:


corr_dauer = df_input_pre[list_non_txt_cols].drop("DAUER_KAT_num", axis=1).apply(lambda x: x.corr(df_input_pre[list_non_txt_cols].DAUER_KAT_num))


# In[1192]:


corr_dauer


# In[1195]:


fig = plt.figure(figsize = (30, 15))
sns.heatmap(corr_non_text)


# In[1196]:


corr_non_text


# In[ ]:




