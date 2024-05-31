# 21207295 Yiran Zhao

import sys
import argparse
import logging
import math
import os
import string
import time

STOPWORDS_PATH = './files/stopwords.txt'
QUERY_PATH = './files/queries.txt'
UCD_NUMBER = '21207295'


def get_args() -> argparse.Namespace:
    """
    Get the command line arguments
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        type=str,
                        help='Path to comp3009j-corpus-large')
    parser.add_argument('-m',
                        type=str,
                        choices=['interactive', 'automatic'],
                        help='mode, either "interactive" or "automatic"')
    args = parser.parse_args()
    while args.p is None or not os.path.exists(args.p):
        if args.p is not None:
            print(f"Path {args.p} does not exist. Please enter a valid path.")
        args.p = input("Enter path to comp3009j-corpus-large: ")
    while args.m is None or args.m not in ['interactive', 'automatic']:
        print(f"Mode {args.m} is not valid. Please enter a valid mode.")
        args.m = input("Enter mode (either 'interactive' or 'automatic'): ")
    return args


def load_index() -> dict:
    """
    Load the BM25 index from index file
    :return: dictionary of BM25 index
    """
    logging.info(f'Loading BM25 index from file {UCD_NUMBER}-large.index, please wait.')
    index = {}
    with open(f'{UCD_NUMBER}-large.index', encoding='utf-8', mode='r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Total number of documents in the collection that contain term'):
                terms_str = line.split('\t')[1]
                terms = terms_str.split(', ')
                index['docs_terms'] = {term.split('(')[0]: int(term.split('(')[1].replace(')', '')) for term in terms}
            elif line.startswith('The average length of a document in the corpus'):
                index['avg_doc_length'] = float(line.split('\t')[1])
            elif line.startswith('Total number of documents in the collection'):
                index['N'] = int(line.split('\t')[1])
            else:
                docID, length, terms_str = line.split('\t')
                if length == '0' or terms_str == 'None':
                    index[docID] = {'length': int(length), 'terms': {}}
                else:
                    terms = terms_str.split(', ')
                    index[docID] = {'length': int(length),
                                    'terms': {term.split('(')[0]: int(term.split('(')[1].replace(')', '')) for term in
                                              terms if '(' in term}}
    logging.info(f'BM25 index loaded successfully.')
    return index


def get_docs_dict(index: dict) -> dict:
    """
    Get the dictionary of each documents' term and frequency
    :param index: BM25 index
    :return: dictionary of documents' term and frequency
    """
    docs_term = {}
    for docID, doc_info in index.items():
        if isinstance(doc_info, dict) and "terms" in doc_info:
            docs_term[docID] = doc_info
    return docs_term


def preprocess_query(query: str) -> list[str]:
    """
    Preprocess the query
    :param query: query string
    :return: list of preprocessed terms
    """
    terms = query.lower()
    terms = terms.replace('-', ' ')
    terms = terms.translate(str.maketrans('', '', string.punctuation))
    terms = terms.strip().split()
    terms = [porter.PorterStemmer().stem(term) for term in terms if term not in stopwords]
    return terms


def read_queries(query_path: str) -> dict[str, list[str]]:
    """
    Read the queries from the queries file
    :param query_path: path to the queries file
    :return: list of queries
    """
    while not os.path.exists(query_path):
        print(f"Queries path {query_path} does not exist. Please enter a valid path.")
        query_path = input("Enter path to the queries file: ")
    queries_terms = {}
    with open(query_path, encoding='utf-8', mode='r') as f:
        queries_file = [line.strip().split(' ', 1) for line in f]
        for query in queries_file:
            terms = preprocess_query(query[1])
            queries_terms[query[0]] = terms
        logging.info(f'Successfully read queries from {query_path}')
    return queries_terms


def read_stopwords(stopwords_path: str) -> list[str]:
    """
    Read the stopwords from the stopwords file
    :param stopwords_path: path to the stopwords file
    :return: list of stopwords
    """
    while not os.path.exists(stopwords_path):
        print(f"Stopwords path {stopwords_path} does not exist. Please enter a valid path.")
        stopwords_path = input("Enter path to the stopwords file: ")
    with open(stopwords_path, encoding='utf-8', mode='r') as f:
        stopwords_file = f.read().strip().split()
        logging.info(f'Successfully read stopwords from {stopwords_path}')
        return stopwords_file


def calculate_scores(query_terms: list, docs_dict: dict, docs_terms: dict, N: int, avg_doc_length: float) -> dict:
    """
    Calculate the BM25 scores for the query
    :param query_terms: list of query terms
    :param docs_dict: dictionary of frequency of terms in documents
    :param docs_terms: dictionary of total number of documents in the collection that contain term
    :param N: Total number of documents in the collection
    :param avg_doc_length: The average length of a document in the corpus
    :return:
    """
    k = 1
    b = 0.75
    scores = {}
    for docID, doc_info in docs_dict.items():
        length = doc_info['length']
        terms = doc_info['terms']
        total_score = 0
        for term in query_terms:
            if term in terms:
                f = terms[term]
            else:
                f = 0
            if term in docs_terms:
                idf = math.log2((N - docs_terms[term] + 0.5) / (docs_terms[term] + 0.5))
            else:
                idf = math.log((N + 0.5) / 0.5)
            score = idf * ((f * (k + 1)) / (f + k * (1 - b + b * length / avg_doc_length)))
            total_score += score
        scores[docID] = total_score
    # logging.info(f'Successfully calculated BM25 scores for query {query_terms}')
    return scores


def interactive_mode(index: dict) -> None:
    docs_dict = get_docs_dict(index)
    N = index['N']
    avg_doc_length = index['avg_doc_length']
    docs_terms = index['docs_terms']
    while True:
        query = input('Enter a query: ')
        if query == 'QUIT':
            break
        query_terms = preprocess_query(query)
        scores = calculate_scores(query_terms, docs_dict, docs_terms, N, avg_doc_length)
        print(f'Results for query [{query}]')
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for i, (docID, score) in enumerate(sorted_scores[:15]):
            print(f'{i + 1} {docID} {score}')


def automatic_mode(query_file: dict, index: dict) -> dict:
    docs_dict = get_docs_dict(index)
    N = index['N']
    avg_doc_length = index['avg_doc_length']
    docs_terms = index['docs_terms']
    query_scores = {}
    for qID, query in query_file.items():
        scores = calculate_scores(query, docs_dict, docs_terms, N, avg_doc_length)
        query_scores[qID] = scores
    return query_scores


def save_results(score: dict[str, dict[str, float]]) -> None:
    """
    Save the results to a result file

    Note: Only the top 40 results are saved for each query
    I chose to save the top 40 results because evaluations showed that this approach yielded relatively higher scores.

    :param score: dictionary of scores
    :return: None
    """
    with open(f'{UCD_NUMBER}-large.results', encoding='utf-8', mode='w') as f:
        for qID, scores in score.items():
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for i, (docID, score) in enumerate(sorted_scores[:40]):
                f.write(f'{qID} {docID} {i + 1} {score}\n')
            # for i, (docID, score) in enumerate(scores):
            #     if score > 0:
            #         f.write(f'{qID} {docID} {i + 41} {score}\n')
    logging.info(f'Successfully saved index to {UCD_NUMBER}-large.results')


if __name__ == '__main__':
    start_time = time.process_time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    logging.info(f'Path to comp3009j-corpus-large: {args.p}')
    logging.info(f'Mode: {args.m}')
    sys.path.append(os.path.dirname(args.p))
    from files import porter
    QUERY_PATH = os.path.join(args.p, 'files/queries.txt')
    STOPWORDS_PATH = os.path.join(args.p, 'files/stopwords.txt')
    stopwords = read_stopwords(STOPWORDS_PATH)
    index = load_index()
    if args.m == 'interactive':
        interactive_mode(index)
    elif args.m == 'automatic':
        queries = read_queries(QUERY_PATH)
        results = automatic_mode(queries, index)
        save_results(results)
    end_time = time.process_time()
    print('Time is {} seconds'.format( end_time - start_time))
