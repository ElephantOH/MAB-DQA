import argparse
import json
import numpy as np
from math import log2
import os
import prettytable as pt


def ndcg_cell(ground_truth, prediction, k):
    k = min(len(prediction), k)
    if k == 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(prediction[:k]):
        rel = 1.0 if doc_id in ground_truth else 0.0
        dcg += rel / log2(i + 2)

    num_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(k))

    if idcg == 0:
        return 0.0

    return dcg / idcg * 100.0

def mrr_cell(ground_truth, prediction, k):
    for i, item in enumerate(prediction[:k]):
        if item in ground_truth:
            return (1.0 / (i + 1)) * 100.0
    return 0.0


def f1_score_cell(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_rag_one_sample(support_context, pred_context, top_k=[1, 3, 5, 10]):

    metrics = {}

    len_support_context = len(support_context)

    for k in top_k:
        cur_pred = pred_context[:k]
        intersect = len(set(cur_pred) & set(support_context))

        recall = intersect / len_support_context * 100.0
        precision = intersect / len(cur_pred) * 100.0

        metrics[f"recall@{k}"] = recall
        metrics[f"precision@{k}"] = precision
        metrics[f"f1@{k}"] = f1_score_cell(precision, recall)
        metrics[f"irrelevant@{k}"] = (len(cur_pred) - intersect) / len(cur_pred) * 100.0

        metrics[f"ndcg@{k}"] = ndcg_cell(support_context, cur_pred, k)
        metrics[f"mrr@{k}"] = mrr_cell(support_context, cur_pred, k)

    return metrics


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # only MMLong and LongDocURL provide the ground-truth evidence pages
    args.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL"])
    args.add_argument("--k_list", type=list, default=[1, 2, 3, 4, 5, 10])
    args = args.parse_args()


    filepaths = [
        # "data/LongDocURL/samples-retrieval_colpali.json",
        # "data/LongDocURL/samples-retrieval_colpali_clip.json",
        # "data/LongDocURL/samples-retrieval_molo.json",
        # "data/LongDocURL/samples-retrieval_hydragraph.json",
        # "data/LongDocURL/samples-retrieval_qsp_34.json",
        "datasets/MMLongBench/samples[retrieval_colpali].json",
        "datasets/MMLongBench/samples[retrieval_mab].json",
        # "data/MMLongBench/sample-with-retrieval-results.json",
        # "data/MMLongBench/samples-retrieval_colpali.json",
        # "data/MMLongBench/samples-retrieval_molo.json",
        # "data/MMLongBench/samples-retrieval_molo_plus.json",

        "datasets/MMLongBench/samples-retrieval_qsp_12.json",
        # "data/MMLongBench/samples-retrieval_qsp_ts_lambda_0.95_p.json",

        # "data/MMLongBench/samples-retrieval_molo_llava.json",
        # "data/MMLongBench/samples-retrieval_molo_32b.json",
        # "data/MMLongBench/samples-retrieval_qsp_nf_32b.json",
    ]

    for filepath in filepaths:
        if os.path.exists(filepath):
            samples = json.load(open(filepath, 'r'))

            table = pt.PrettyTable()
            table.field_names = ["Method", "K", "Recall(%)", "Precision(%)", "F1(%)", "NDCG(%)", "MRR(%)",
                                 "Irrelevant(%)"]
            for target in ["text", "image"]:

                all_metrics = {f'recall@{k}': [] for k in args.k_list}
                for remain_metric in ['ndcg', 'mrr', 'precision', 'f1', 'irrelevant']:
                    all_metrics.update({f'{remain_metric}@{k}': [] for k in args.k_list})

                for sample in samples:
                    ground_truth = sample["evidence_pages"]
                    if ground_truth == []:
                        continue

                    if isinstance(ground_truth, str):
                        try:
                            ground_truth = [int(page.strip()) for page in
                                            ground_truth.strip('[]').split(',')] if ground_truth != "[]" else []
                        except (ValueError, AttributeError):
                            print("Error parsing evidence_pages string")
                            ground_truth = []
                    else:
    
                        ground_truth = list(ground_truth)

                    import re

                    # pattern = re.compile(rf"{re.escape(target)}-top-\d+-question")
                    pattern = re.compile(r"retrieval\[[^]]+\]")
                    matched_keys = [key for key in sample.keys() if pattern.match(key)]

                    if not matched_keys:
                        continue

                    matched_key = matched_keys[0]

                    k_match = re.search(r"-top-(\d+)-question", matched_key)
                    if k_match:
                        k = int(k_match.group(1))
                    else:
                        k = 10

                    cur_preds = sample[matched_key]

                    if len(cur_preds) < k:
                        cur_preds.extend([-1] * (k - len(cur_preds)))

                    if ground_truth == []:
                        continue
                    else:
                        ground_truth = [g - 1 if g > 0 else g for g in ground_truth]

                    scores = evaluate_rag_one_sample(support_context=ground_truth,
                                                     pred_context=cur_preds,
                                                     top_k=args.k_list)

                    for metric_name, value in scores.items():
                        all_metrics[metric_name].append(value)

                for metric_name, values in all_metrics.items():
                    all_metrics[metric_name] = np.round(np.mean(values), 2)

                for k in args.k_list:
                    table.add_row([
                        f"mdocagent-{target}",
                        k,
                        all_metrics[f'recall@{k}'],
                        all_metrics[f'precision@{k}'],
                        all_metrics[f'f1@{k}'],
                        all_metrics[f'ndcg@{k}'],
                        all_metrics[f'mrr@{k}'],
                        all_metrics[f'irrelevant@{k}']
                    ])
            print(filepath)
            print(table, '\n')