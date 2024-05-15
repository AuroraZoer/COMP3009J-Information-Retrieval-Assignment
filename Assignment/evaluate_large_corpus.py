# 21207295 Yiran Zhao

import argparse
import logging
import math
import os

QRELS_PATH = './files/qrels.txt'
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
    args = parser.parse_args()
    while args.p is None or not os.path.exists(args.p):
        if args.p is not None:
            print(f"Path {args.p} does not exist. Please enter a valid path.")
        args.p = input("Enter path to comp3009j-corpus-large: ")
    return args


def read_qrels(qrels_path: str) -> dict[str, dict[str, int]]:
    """
    Read the qrels from the qrels file
    :param qrels_path: path to the qrels file
    :return: dictionary of qrels
    """
    qrels = {}
    while not os.path.exists(qrels_path):
        print(f"Qrels path {qrels_path} does not exist. Please enter a valid path.")
        qrels_path = input("Enter path to the qrels file: ")
    with open(qrels_path, encoding='utf-8', mode='r') as f:
        for line in f:
            qid, ignored, docID, rel = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docID] = int(rel)
    logging.info(f'Successfully read qrels from {qrels_path}')
    return qrels


def load_results() -> dict[str, list[tuple[str, int, float]]]:
    """
    Load the results from the results file
    :return: dictionary of results, where the key is the query ID and the value is a list of tuples
    """
    logging.info(f'Loading results from file {UCD_NUMBER}.results, please wait.')
    results = {}
    with open(f'{UCD_NUMBER}-large.results', encoding='utf-8', mode='r') as f:
        for line in f:
            qid, docID, rank, score = line.strip().split()
            if qid not in results:
                results[qid] = []
            # Append the tuple (docID, rank, score) to the list of results for the query ID
            # use tuple because we want to keep the order of the elements, and we don't want to change them
            results[qid].append((docID, int(rank), float(score)))
    logging.info(f'Results loaded successfully.')
    return results


def evaluation(qrels: dict, results: dict) -> tuple:
    """
    Evaluate the results using the qrels
    :param qrels: the qrels dictionary
    :param results: the results dictionary
    :return: the metrics
    """
    precision = 0
    recall = 0
    r_precision = 0
    precision_at_15 = 0
    map = 0
    ndcg_at_15 = 0
    bpref = 0
    for qID in qrels:
        relevant = qrels[qID]
        retrieved = results[qID]
        rel = list(relevant.keys())
        ret = [docID for docID, _, _ in retrieved]
        relret = set(ret) & set(rel)
        precision += len(relret) / len(ret)
        recall += len(relret) / len(rel)

        relevant_in_top_15 = set(ret[:15]) & set(rel)
        precision_at_15 += len(relevant_in_top_15) / 15
        relevant_in_top_r = set(ret[:len(rel)]) & set(rel)
        r_precision += len(relevant_in_top_r) / len(rel)

        # MAP
        average_precision = 0
        num_relevant = 0
        for i, docID in enumerate(ret):
            if docID in rel:
                num_relevant += 1
                average_precision += num_relevant / (i + 1)
        map += average_precision / len(rel)

        # NDCG@15
        DCG, IDCG, NDCG = [], [], []
        sorted_relevant = sorted(relevant.items(), key=lambda x: x[1], reverse=True)
        for i, docID in enumerate(ret):
            # Calculate DCG
            if docID in relevant.keys():
                relevance = relevant[docID]
                if i == 0:  # rank = 1
                    DCG.append(relevance)
                else:
                    DCG.append(relevance / math.log2(i + 1) + DCG[i - 1])
            else:
                if i == 0:
                    DCG.append(0)
                else:
                    DCG.append(DCG[i - 1])
            # Calculate IDCG
            if i < len(sorted_relevant):
                relevance = sorted_relevant[i][1]
            else:
                relevance = 0
            if i == 0:
                IDCG.append(relevance)
            else:
                IDCG.append(relevance / math.log2(i + 1) + IDCG[i - 1])
            # Calculate NDCG
            NDCG.append(DCG[i] / IDCG[i])
        ndcg_at_15 += NDCG[14]

        # bpref
        R = len(rel)
        non_rel_num = 0  # number of non-relevant documents
        bpref_sum = 0  # sum of bpref scores
        for docID in ret:
            # stop when the number of non-relevant documents is higher than the number of relevant documents
            if non_rel_num >= R:
                break
            # record if the document is non-relevant
            if docID not in rel:
                non_rel_num += 1
            # add to bpref_sum if the document is relevant
            else:
                bpref_sum += 1 - non_rel_num / R
        bpref += bpref_sum / R if R != 0 else 0
    return precision, recall, r_precision, precision_at_15, ndcg_at_15, map, bpref


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    logging.info(f'Path to comp3009j-corpus-large: {args.p}')
    QRELS_PATH = os.path.join(args.p, 'files/qrels.txt')
    qrels = read_qrels(QRELS_PATH)
    results = load_results()
    precision, recall, r_precision, precision_at_15, ndcg_at_15, map, bpref = evaluation(qrels, results)
    print(f'Evaluation results:')
    print("Precision:   {:.3f}".format(precision / len(qrels)))
    print("Recall:      {:.3f}".format(recall / len(qrels)))
    print("R-Precision: {:.3f}".format(r_precision / len(qrels)))
    print("P@15:        {:.3f}".format(precision_at_15 / len(qrels)))
    print("NDCG@15:     {:.3f}".format(ndcg_at_15 / len(qrels)))
    print("MAP:         {:.3f}".format(map / len(qrels)))
    print("bpref:       {:.3f}".format(bpref / len(qrels)))
