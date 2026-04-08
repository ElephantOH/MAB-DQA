import argparse
import json
import numpy as np
import os
import prettytable as pt

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL"])
    args.add_argument("--k_list", type=list, default=[1, 2, 3, 4, 5, 10])
    args = args.parse_args()
    file_sample_counts = []
    filepaths = [
        # "data/MMLongBench/samples-retrieval_molo-qa_simple_vis4.json",
        # "data/MMLongBench/samples-retrieval_qsp_nf-qa_multi_vis4.json",
        # "data/MMLongBench/samples-retrieval_qsp_nf-qa_simple_vis4.json",
        "data/MMLongBench/samples-retrieval_qsp_nf.json",
        "data/LongDocURL/samples-retrieval_molo.json",
        "data/FetaTab/samples-retrieval_qsp_nf.json",
        "data/PaperTab/samples-retrieval_molo-qa_simple_vis4.json",
    ]

    for filepath in filepaths:
        if os.path.exists(filepath):
            samples = json.load(open(filepath, 'r'))

            table = pt.PrettyTable()
            table.field_names = ["Method", "Binary Correctness(%)"]
            file_sample_count = len(samples)

            file_sample_counts.append((filepath, file_sample_count))
            print(filepath, file_sample_count)

  
            binary_correctness_scores = []
            eval_openai_scores = []

            for sample in samples:
                if "binary_correctness" in sample:
                    binary_correctness_scores.append(sample["binary_correctness"])
                if "eval_openai" in sample:
                    eval_openai_scores.append(sample["eval_openai"])

            if binary_correctness_scores:
                avg_binary_correctness = np.round(np.mean(binary_correctness_scores) * 100, 2)
            else:
                avg_binary_correctness = -1

            if eval_openai_scores:
                avg_eval_openai = np.round(np.mean(eval_openai_scores) * 100, 2)
            else:
                avg_eval_openai = -1

            table.add_row([os.path.basename(filepath), avg_binary_correctness])
            table.add_row([os.path.basename(filepath), avg_eval_openai])

            print(filepath)
            print(table, '\n')