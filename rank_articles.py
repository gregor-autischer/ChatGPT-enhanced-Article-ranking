#!/usr/bin/env python3

import sys
import openai
from openai import OpenAI
import json
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import generate_html, create_plot, check_api_key, plot_absolute_error_by_index, plot_precision_recall_curve

DATA_FOLDER = "data"
RAW_DATASET = "dataset.jsonl"
PREPROCESSED_DATASET = "preprocessed_dataset.jsonl"

PLOTS_FOLDER = "plots"

HTML_FOLDER = "html"

"""Extending the Search Term with GPT-4o"""
def extend_search_term_with_gpt(search_term_string):
    # ChatGPT integration
    
    # ----------- Change this to test different prompts ------------
    # Sets the behaviour or context of the model
    gpt_system_content = "You find additional synonymes and words to improve a news article search"
    # Is the user input to the model (like the chat message in ChatGPT)
    gpt_user_content = f"Give me additional words and synonyms that improve a keyword search for {search_term_string}. Just return between 5 and 10 words as a string with no bulletpoints or similar. Return only words no phrases."
    # ----------- ------------------------------------- ------------

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
        model="gpt-4o",
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

"""Preprocessing the Data: removing punctuation, make everything lowercase, stopwords, and stemming"""
def preprocess(content):
    # Remove punctuation and lowercase
    content = re.sub(r'[^\w\s]', '', content.lower())
    
    # Tokenize the content
    tokens = word_tokenize(content)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens] 

    return ' '.join(tokens)

"""Removes Chinese, Japanese, and Korean characters"""
def remove_CJK_characters(content):
    # Remove Chinese, Japanese and Korean characters
    content = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\u1100-\u11ff\uac00-\ud7af]', '', content)

    return content

"""Preprocess the data and save it to a new file"""
def preprocess_and_save(input_path, output_path):
    print("Preprocessing data...")
    documents = []
    with open(input_path, 'r') as infile:
        for line in infile:
            doc = json.loads(line)
            # Step 1: Remove Chinese, Japanese and Korean characters
            doc['content_abc'] = remove_CJK_characters(doc['content'])
            doc['title_abc'] = remove_CJK_characters(doc['title'])
            
            # Step 2: Preprocess both the cleaned content and title
            processed_content = preprocess(doc['content_abc'])  # Preprocessed content
            processed_title = preprocess(doc['title_abc'])  # Preprocessed title
            
            # Step 3: Add preprocessed fields to the document
            doc['processed_content'] = processed_content
            doc['processed_title'] = processed_title
            
            # Step 4: Combine processed content and title
            doc['processed_combined'] = f"{processed_title} {processed_content}"
            
            documents.append(doc)

    with open(output_path, 'w') as outfile:
        for doc in documents:
            json.dump(doc, outfile)
            outfile.write('\n')  # Ensure newline-separated JSON

"""Determine the relevance"""
def determine_relevance(df, query, use_extended_query=False):
    query_terms = set(query.split())  # Tokenize the query

    # Check if any query term appears in the document content
    def is_relevant(content):
        content_tokens = set(preprocess(content).split())  # Tokenize the content
        return 1 if query_terms & content_tokens else 0  # Intersection check

    return df['content'].apply(is_relevant)

"""Main function: runs the ranking of the articles using BM25"""
def model_bm25(preprocessed_file):
    documents = []
    with open(preprocessed_file, 'r') as f:
        for line in f:
            documents.append(json.loads(line))

    df = pd.DataFrame(documents)

    print("Below is the overview of the preprocessed data")
    print(df.head())

    # Get tokenized text (title + content)
    tokenized_corpus = [doc.split() for doc in df['processed_combined']]
    bm25 = BM25Okapi(tokenized_corpus)

    search_term = input("Please enter your search term: ")
    search_term_extended = extend_search_term_with_gpt(search_term)
    print(f"Original Query: {search_term}")
    print(f"Expanded Query: {search_term_extended}")

    expanded_query_processed = preprocess(search_term_extended)
    expanded_query_tokens = expanded_query_processed.split()

    query_processed = preprocess(search_term)
    query_tokens = query_processed.split()

    # Rank documents with expanded query
    scores = bm25.get_scores(expanded_query_tokens)
    df['bm25_score_extended_query'] = scores

    # Get top results with expanded query
    top_results_extended_query = df.sort_values(by='bm25_score_extended_query', ascending=False).head(10)
    print("Top results with expanded query:")
    print(top_results_extended_query[['title_abc', 'bm25_score_extended_query']])

    # Rank documents with simple query
    scores = bm25.get_scores(query_tokens)
    df['bm25_score_simple_query'] = scores

    # Get top results with simple query
    top_results_simple_query = df.sort_values(by='bm25_score_simple_query', ascending=False).head(10)
    print("Top results with simple query:")
    print(top_results_simple_query[['title_abc', 'bm25_score_simple_query']])

    # Generate HTML for top results
    generate_html(top_results_simple_query, top_results_extended_query, "top_results.html", HTML_FOLDER)

    # Create 'relevant' column based on your expaned_query
    df['relevant'] = determine_relevance(df, expanded_query_processed, use_extended_query=False)

    # Evaluate the search results
    evaluate(df, top_results_extended_query, expanded_query=True)
    evaluate(df, top_results_simple_query, expanded_query=False)

"""Evaluate the search results"""
def evaluate(df, top_results, expanded_query):
    # ground truth relevance score
    y_true = df['relevant'].values

    # Predicted BM25 scores (use the right column)
    score_column = "bm25_score_extended_query" if expanded_query else "bm25_score_simple_query"
    y_pred = df[score_column].values

    # compute metrics
    precision = precision_score(y_true, y_pred >= 0.5)
    recall = recall_score(y_true, y_pred >= 0.5)
    f1 = f1_score(y_true, y_pred >= 0.5)

    print("Ground Truth (y_true):", np.unique(y_true))
    print("Predicted Scores (y_pred):", np.min(y_pred), np.max(y_pred))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100 # never divide by zero!

    print(f"Evaluation for {'Extended Query' if expanded_query else 'Simple Query'}:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    # Define folder and file name
    PLOTS_FOLDER = "output_plots"
    os.makedirs(PLOTS_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
    plot_file_name = "top_ranked_results_expanded_query.png" if expanded_query else "top_ranked_results_simple_query.png"
    plot_file_path = os.path.join(PLOTS_FOLDER, plot_file_name)

    # Use the correct column for bm25_score based on query type
    query_type = "Extended Query" if expanded_query else "Simple Query"

    # Create the plot using the refactored function
    create_plot(top_results, score_column, query_type, plot_file_path)

    plot_file_path = os.path.join(PLOTS_FOLDER, "precisionc_recall_curve_expanded_query.png" if expanded_query else "precisionc_recall_curve_simple_query.png")
    plot_precision_recall_curve(y_true, y_pred, query_type, plot_file_path)

    plot_file_path = os.path.join(PLOTS_FOLDER, "plot_absolut_error_expanded_query.png" if expanded_query else "plot_absolut_error_simple_query.png")
    plot_absolute_error_by_index(y_true, y_pred, query_type, plot_file_path)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')

    if len(sys.argv) != 2:  # Check if an argument is provided
        print("Usage: python script.py <preprocess|search>")
        sys.exit(1)

    argument = sys.argv[1].lower()  # Get the argument and convert it to lowercase

    if argument == "preprocess":
        raw_data_path = os.path.join(DATA_FOLDER, RAW_DATASET)
        pp_data_path = os.path.join(DATA_FOLDER, PREPROCESSED_DATASET)
        preprocess_and_save(raw_data_path, pp_data_path)
        print(f"Data preprocessed and saved to {pp_data_path}")

    elif argument == "search":
        check_api_key()  # Ensure API key is set
        pp_data_path = os.path.join(DATA_FOLDER, PREPROCESSED_DATASET)
        model_bm25(pp_data_path)

    else:
        print("Usage: python script.py <preprocess|search>")
        sys.exit(1)
    
    