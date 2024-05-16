# 21207295 Yiran Zhao
import sys
import argparse
import logging
import os
import re
import string

DOCUMENT_PATH = './documents'
STOPWORDS_PATH = './files/stopwords.txt'
UCD_NUMBER = '21207295'


def get_args() -> argparse.Namespace:
    """
    Get the command line arguments
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        type=str,
                        help='Path to comp3009j-corpus-small')
    args = parser.parse_args()
    while args.p is None or not os.path.exists(args.p):
        if args.p is not None:
            print(f"Path {args.p} does not exist. Please enter a valid path.")
        args.p = input("Enter path to comp3009j-corpus-small: ")
    return args


def read_documents(documents_path: str) -> dict[str, str]:
    """
    Read the documents from the documents directory
    :param documents_path: path to the documents directory
    :return: dictionary of documents
    """
    docs = {}
    while not os.path.exists(documents_path):
        print(f"Documents path {documents_path} does not exist. Please enter a valid path.")
        documents_path = input("Enter path to the documents directory: ")
    for filename in os.listdir(documents_path):
        if filename != '.DS_Store':  # Ignore macOS .DS_Store files
            with open(os.path.join(documents_path, filename), encoding='utf-8', mode='r') as f:
                docs[filename] = f.read()
    logging.info(f'Successfully read documents under {documents_path}')
    return docs


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


def process_document(documents: dict[str, str], stopwords: list[str]) -> dict[str, list[str]]:
    """
    Process the documents by removing punctuation, converting to lowercase, removing stopwords and stemming

    Note: I evaluated two preprocessing scenarios: replacing hyphens with spaces and completely removing hyphens.
    The evaluation results showed that replacing hyphens with spaces yielded higher scores, so I chose this approach for preprocessing.

    :param documents: dictionary of documents
    :param stopwords: list of stopwords
    :return: dictionary of documents with processed terms
    """
    stemmer = porter.PorterStemmer()
    docs_dict = {}
    i = 0
    for docID, doc in documents.items():
        doc = doc.lower()
        # The following regex is used to remove [ ... .gif/png/jpg] from the document.
        # I consider that textual information embedded in such image formats is considered to provide negligible utility
        # in the context of a search operation because of their typically non-semantic and context-independent nature.
        doc = re.sub(r'\[.*\.(gif|png|jpg)\]', ' ', doc)
        # Replace hyphens with spaces before removing punctuation
        doc = doc.replace('-', ' ')
        # Remove punctuation
        doc = doc.translate(str.maketrans('', '', string.punctuation))
        # Split the document into terms
        terms = doc.strip().split()
        # Perform stopword removal and stemming
        terms = [stemmer.stem(term) for term in terms if term not in stopwords]
        docs_dict[docID] = terms

        i += 1
        print(f'Processed document {i}/{len(documents)}', end='\r')

    logging.info('Successfully processed documents')
    return docs_dict


def create_index(docs_dict: dict[str, list[str]]) -> dict:
    """
    Create the index from the processed documents
    :param docs_dict: dictionary of documents with processed terms
    :param documents: dictionary of documents
    :return: index
    """
    index = {}
    index["N"] = len(docs_dict)
    sum_length = 0
    docs_terms = {}
    for docID, terms in docs_dict.items():
        terms_in_doc = {}
        seen_terms = set()
        doc_length = len(terms)
        for term in terms:
            if term not in terms_in_doc:
                terms_in_doc[term] = 1
            else:
                terms_in_doc[term] += 1
            if term not in seen_terms:
                seen_terms.add(term)
                if term not in docs_terms:
                    docs_terms[term] = 1
                else:
                    docs_terms[term] += 1
        sum_length += doc_length
        index[docID] = {'terms': terms_in_doc, 'length': doc_length}
    avg_doc_len = sum_length / len(docs_dict)
    index['avg_doc_length'] = avg_doc_len
    index['docs_terms'] = docs_terms
    logging.info('Successfully created index')
    return index


def save_index(index: dict) -> None:
    """
    Save the index to a file
    :param index:
    :return: None
    """
    with open(f'{UCD_NUMBER}-small.index', encoding='utf-8', mode='w') as f:
        for docID, doc_info in index.items():
            if docID == 'docs_terms':
                terms_str = ', '.join(
                    f'{k}({v})' for k, v in sorted(doc_info.items(), key=lambda item: item[1], reverse=True))
                f.write(f'Total number of documents in the collection that contain term\t{terms_str}\n')
            elif docID == 'avg_doc_length':
                f.write(f'The average length of a document in the corpus\t{doc_info}\n')
            elif docID == 'N':
                f.write(f'Total number of documents in the collection\t{doc_info}\n')
            elif isinstance(doc_info, dict):
                if doc_info["terms"]:
                    terms_str = ', '.join(
                        f'{k}({v})' for k, v in
                        sorted(doc_info["terms"].items(), key=lambda item: item[1], reverse=True))
                else:
                    terms_str = 'None'
                f.write(
                    f'{docID}\t{doc_info["length"]}\t{terms_str}\n')
    logging.info(f'Successfully saved index to {UCD_NUMBER}.index')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    logging.info(f'Path to comp3009j-corpus-small: {args.p}')
    sys.path.append(os.path.dirname(args.p))
    from files import porter
    DOCUMENT_PATH = os.path.join(args.p, 'documents')
    STOPWORDS_PATH = os.path.join(args.p, 'files/stopwords.txt')
    documents = read_documents(DOCUMENT_PATH)
    stopwords = read_stopwords(STOPWORDS_PATH)
    docs_terms_dict = process_document(documents, stopwords)
    index = create_index(docs_terms_dict)
    save_index(index)
