from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
import spacy
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

def convert_pdfminer(fname):
    """Converts pdf to text"""
    fp = open(fname, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    text = ''
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
        layout = device.get_result()
        for lt_obj in layout:
            if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                text += lt_obj.get_text()
    return text
    
def cleaning_fast(raw_text,stop_words,postags):
    """Allowed postags probably something from the list ['NOUN', 'ADJ', 'VERB', 'ADV']"""
    docs_cleaned = []
    temp = []
    allowed_postags=postags
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    for item in raw_text:
        tokens = gensim.utils.simple_preprocess(item)
        red_tokens = [word for word in tokens if word not in stop_words]
        doc = nlp(" ".join(red_tokens))
        lemmas = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        docs_cleaned.append(lemmas)
    return docs_cleaned

def ldamodel_fast(docs_cleaned,no_pass, no_topics,chunk_divider,filter_up,alp='asymmetric'):
    id2word = corpora.Dictionary(docs_cleaned)
    id2word.filter_extremes(no_below=2, no_above=filter_up, keep_n=50000)
    corpus = [id2word.doc2bow(text) for text in docs_cleaned]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=no_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=len(corpus)/chunk_divider,
                                           passes=no_pass,
                                           alpha=alp,
                                           per_word_topics=False,
                                            minimum_probability = 0.,
                                            eta = 'auto')
    return lda_model,corpus,id2word

def create_top_words_table(lda_model, no_topics, no_words,probs):
    top_words_df = pd.DataFrame()
    for i in range(no_topics):
        temp_words = lda_model.show_topic(i,no_words)
        if probs:
            words = temp_words
        else:
            words = [name for (name,_) in temp_words]
        top_words_df['Topic ' + str(i+1)] = words
    return top_words_df

def topic_evolution_table(lda_model, corpus, file_years, no_topics,titles):
    evolution = np.zeros([len(corpus),no_topics])
    ind = 0
    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        for topic in topics:
            evolution[ind,topic[0]] = topic[1]
        ind+=1
    evolution_df = pd.DataFrame(evolution,columns=titles)
    evolution_df['Year'] = file_years
    evolution_df['Date'] = pd.to_datetime(evolution_df['Year'],format = "%Y")
    evolution_df.set_index('Date',inplace=True)
    evolution_df.drop('Year',axis=1,inplace = True)
    return evolution_df

def optimal_settings(corpus,id2word,docs_cleaned,topic_range,chunk_split_range,no_pass,alp='asymmetric'):
    coh_table = np.zeros([max(topic_range)+1,max(chunk_split_range)+1])
    perp_table = np.zeros([max(topic_range)+1,max(chunk_split_range)+1])
    for i in topic_range:
        for j in chunk_split_range:
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=i,random_state=100,
                                                        update_every=1,chunksize=len(corpus)/j,passes=no_pass,alpha=alp,
                                                        per_word_topics=False,minimum_probability = 0.,eta='auto')
            coherence_model_lda = CoherenceModel(model=lda_model, texts = docs_cleaned,corpus=corpus, dictionary=id2word, coherence='c_v')
            coh_table[i,j] = coherence_model_lda.get_coherence()
            perp_table[i,j] = lda_model.log_perplexity(corpus)
    return coh_table,perp_table

def get_representative_documents(lda_model,corpus,no_topics,file_names):
    doc_topics = []
    for doc in corpus:
        doc_topics.append([item for (_,item) in lda_model.get_document_topics(doc)])
    doc_topics_df = pd.DataFrame(doc_topics)
    topic = ['Topic ' + str(i+1) for i in range(no_topics)]
    fnames = [file_names[i] for i in doc_topics_df.idxmax()]
    repres_df = pd.DataFrame()
    repres_df['Topic'] = topic
    repres_df['Document'] = fnames
    return repres_df

def get_top_topics(lda_model,corpus,filenames):
    top_topic = []
    for doc in corpus:
        test=lda_model.get_document_topics(doc)
        test2 = [item[1] for item in test]
        top_topic.append(test2.index(max(test2))+1)
    top_topic_df = pd.DataFrame()
    top_topic_df['Document'] = filenames
    top_topic_df['Top topic'] = top_topic
    return top_topic_df

def get_tfidf_matrix(corpus,id2word):
    tf_idf_model = TfidfModel(corpus,id2word=id2word)
    temp = tf_idf_model[corpus]
    temp = gensim.matutils.corpus2csc(temp)
    return temp.toarray().transpose()

def build_kmeans_info(work_df,km,words_per_cluster,id2word):
    """Work_df must include the document names and tf-idf clusters IN THE SAME ORDER"""
    work_df['Index'] = work_df[work_df.columns[1]]
    work_df.set_index('Index',inplace=True)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(len(km.cluster_centers_)):
        j=i+1
        print("Cluster %d words:" % j, end='')
        for ind in order_centroids[i, :words_per_cluster]:
            print(' %s' % id2word.id2token[ind],end=',')
        print()
        print()
        print("Cluster %d titles:" % j)
        for title in work_df.loc[i][work_df.columns[0]]:
            print('%s,' % title, end='\n')
        print()
        print()
    
def mds_tfidf_plot(tfidf_matrix, tfidf_clusters, file_names,name_prefix):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    dist = 1 - cosine_similarity(tfidf_matrix)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]
    temp_df = pd.DataFrame(dict(x=xs, y=ys, label=tfidf_clusters, title=file_names))
    groups = temp_df.groupby('label')
    fig, ax = plt.subplots(figsize=(27, 27))
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                label='Cluster '+str(name), mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',        
            which='both',    
            bottom='off',  
            top='off',       
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',        
            which='both',    
            left='off',   
            top='off',       
            labelleft='off')
    ax.legend(numpoints=1)

    for i in range(len(temp_df)):
        ax.text(temp_df.iloc[i]['x'], temp_df.iloc[i]['y'], temp_df.iloc[i]['title'], size=12)      
    plt.savefig(name_prefix + '_MDS.png') #show the plot
    
def ward_cluster_plot(tfidf_matrix, doc_names,name_prefix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    linkage_matrix = ward(dist) 
    fig, ax = plt.subplots(figsize=(20, 30))
    ax = dendrogram(linkage_matrix, orientation="left", labels=doc_names,leaf_font_size=10);
    plt.tick_params(\
                axis= 'x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off')
    plt.tight_layout()
    plt.savefig(name_prefix + '_ward_clusters.png', dpi=200)