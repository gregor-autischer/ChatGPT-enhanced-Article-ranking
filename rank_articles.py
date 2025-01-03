#!/usr/bin/env python3

import sys
import openai
from openai import OpenAI
import json
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score


def user_search_term(term):
    # Check if search term is passed
    #if len(sys.argv) > 1:
    if len(term) > 1:
        search_term_string = "".join(term[1:])
    else:
        print("No Search term passed!")
        return
    
    print("Search Term:", search_term_string)

    # extend search term with ChatGPT
    additional_words_string = extend_search_term_with_gpt(search_term_string)

    print("Additional Words from ChatGPT:")
    print(additional_words_string)

    search_term = search_term_string.split()
    search_term_extended = additional_words_string.split()

    print(search_term)
    print(search_term_extended)

    # TODO: Maybe? filter out filler words. I dont know if required?

    # TODO: use search_term and search_term_extended to rank articles
    

def extend_search_term_with_gpt(search_term_string):
    # ChatGPT integration
    
    #gpt_system_content = "You are a helpful assistant."
    gpt_system_content = "You find additional synonymes and words to improve an article search"
    gpt_user_content = f"Give me additional words and synonyms that improve a keyword search for {search_term_string}. Just return between 5 and 15 words as a string with no bulletpoints or similar. Return only words no phrases."
    #openai.api_key = "your-api-key"
    
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": gpt_system_content},
                {"role": "user", "content": gpt_user_content}
            ]
        )
        additional_words_string = completion.choices[0].message.content
    
    except openai.OpenAIError as e:
        print(f"An OpenAI error occurred: {str(e)}")
        sys.exit("Stopping the program due to error.")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit("Stopping the program due to error.")

    return additional_words_string

def preprocess(content):
    content = re.sub(r'[^\w\s]', '', content.lower())  # Remove punctuation and lowercase

    '''if not os.path.exists(nltk.data.find('tokenizers/punkt')):
        nltk.download('punkt')
    if not os.path.exists(nltk.data.find('tokenizers/stopwords')):
        nltk.download('punkt')
    nltk.download('punkt')
    nltk.download('stopwords')'''

    tokens = word_tokenize(content)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)


def model_bm25():
    data = os.path.join('data', 'dataset.jsonl')

    documents = []
    with open(data, 'r') as f:
        for line in f:
            documents.append(json.loads(line))

    df = pd.DataFrame(documents)
    print("Below is the overview of the initial data:")
    print(df.head())  # check data is read correctly

    df['processed_text'] = df['content'].apply(preprocess) #preprocessed data
    print("Below is the overview of the preprocessed data")
    print(df.head())

    tokenized_corpus = [doc.split() for doc in df['processed_text']] #tokenize
    bm25 = BM25Okapi(tokenized_corpus)

    search_term = input("Please enter your search term: ")
    search_term_extended = extend_search_term_with_gpt(search_term)
    print(f"Original Query: {search_term}")
    print(f"Expanded Query: {search_term_extended}")

    expanded_query_processed = preprocess(search_term_extended)
    query_tokens = expanded_query_processed.split()

    # Rank documents
    scores = bm25.get_scores(query_tokens)
    df['bm25_score'] = scores

    # Get top results
    top_results = df.sort_values(by='bm25_score', ascending=False).head(10)
    print(top_results[['content', 'bm25_score']])

    #evaluate(df, top_results)

def evaluate(df, top_results):
    df['relevant'] = [1 if relevance_condition else 0 for relevance_condition in df['text']]

    # Binary relevance scores for top results
    y_true = df['relevant']
    y_pred = [1 if idx in top_results.index else 0 for idx in df.index]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    #graph to check


    plt.figure(figsize=(10, 6))
    plt.barh(top_results['content'], top_results['bm25_score'], color='skyblue')
    plt.xlabel('BM25 Score')
    plt.ylabel('Document')
    plt.title('Top Ranked Documents')
    plt.show()

def check_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"OPENAI_API_KEY is set: {api_key[:5]}...")  # Print the first few characters for confirmation
    else:
        print("OPENAI_API_KEY is not set.")

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    check_api_key()

    model_bm25()
    
    