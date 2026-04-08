import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
CURRENT_FILE_PATH = os.path.abspath(__file__)

import torch
import hydra
import argparse
from tqdm import tqdm
from src.prompt.prompt_mab import PROMPTS
from src.model.factory import ModelFactory
from src.data.factory import DatasetFactory
from src.until.late_interaction import late_interaction, get_top_page
from src.mab.document_hypergraph import DocumentHypergraph
from src.mab.thompson_sampling import ThompsonSamplingBandit

# %%

def parse_args():

    parser = argparse.ArgumentParser(description="MAB Retrieval Pipeline")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mmlb",
        help="dataset_type"
    )
    parser.add_argument(
        "--mission", 
        type=str, 
        default="close_domain_dqa",
        help="mission_key"
    )
    return parser.parse_args()

# %%
def run_page_embed(
        retriever,
        dataset,
        sample,
        contents,
        batch_size=16,
        use_image_embeds=True,
        save_image_embeds=True,
        is_clip=False
    ):
    if use_image_embeds:
        print(f"[Info] [{CURRENT_FILE_PATH}] Loading Precomputed Image Embeddings for Document.")
        image_embeds = dataset.load_image_embeds(sample, is_clip=is_clip)
        if image_embeds is not None:
            return image_embeds
        
    images = [content.image for content in contents]

    image_embeds = retriever.encode_images(
        images=images,
        batch_size=batch_size
    )

    if save_image_embeds:
        dataset.save_image_embeds(sample, image_embeds, is_clip=is_clip)
        print(f"[Info] [{CURRENT_FILE_PATH}] Saved Image Embeddings for Document.")

    return image_embeds

def run_query_embed(
        retriever,
        vlm,
        sample,
        question_key="question",

    ):
    assert question_key in sample, f"[Error] [{CURRENT_FILE_PATH}] Question Key '{question_key}' not Found in Sample."

    initial_query = sample[question_key]
    sample["initial_query"] = initial_query
    basis_queries, _ = vlm.predict(
            question=PROMPTS["query"],
            texts=[initial_query],
            images=None,
            max_new_tokens=60,
        )
    sample["basis_queries"] = basis_queries

    queries = [item.strip() for item in basis_queries.split(',')]
    queries.extend([basis_queries, initial_query])
    query_embeds = retriever.encode_queries(queries)
    return queries, query_embeds

def run_late_interaction(
        page_embeds,
        query_embeds,
        top_k=10,
):
    scores, col_score_dict = late_interaction(page_embeds, query_embeds)
    top_page = get_top_page(scores, top_k)

    top_page_indices_list = []
    top_page_scores_list = []
    for i in range(scores.shape[0]):
        indices = top_page.indices.tolist()[i] if top_page is not None else []
        scores_i = top_page.values.tolist()[i] if top_page is not None else []
        top_page_indices_list.append(indices)
        top_page_scores_list.append(scores_i)

    return top_page_indices_list, top_page_scores_list, col_score_dict, scores

#%%
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config):
    args = parse_args()

    retrival_name = "mab"
    sample_name = "samples"
    retrieval_device = f"cuda:{0}"
    vlm_device = f"auto"
    resume = True

    batch_size = 16
    use_image_embeds = True
    save_image_embeds = True
    is_clip = True
    question_key = "question"
    retrieval_key = f"retrieval[{retrival_name}]"
    top_k = 10

    retriever = ModelFactory.create(
        model_type="colpali/colpali-1.3",
        config=config,
        device_map=retrieval_device,
        dtype="bfloat16",
    )

    vlm = ModelFactory.create(
        model_type="qwen/qwen25vl-7b",
        config=config,
        device_map=vlm_device,
        dtype="bfloat16",
    )

    dataset = DatasetFactory.create(
        mission_key=args.mission,
        dataset_type=args.dataset,
        config=config,
        question_key=question_key,
        retrieval_key=retrieval_key,
    )

    if resume:
        all_samples = dataset.load_samples_file(sample_name=sample_name, suffix=f"[retrieval_{retrival_name}]")
    else:
        all_samples = dataset.load_samples_file(sample_name=sample_name)

    total_samples = len(all_samples)
    pbar = tqdm(total=total_samples, desc="Processing samples")

    doucument_samples = dataset.group_samples_by_doc_id(all_samples)

    for doc_id, samples in doucument_samples.items():

        all_processed = True
        for sample in samples:
            if (retrieval_key not in sample):
                all_processed = False
                break

        if all_processed:
            print(f"[Info] [{CURRENT_FILE_PATH}] Document {doc_id} already Processed, Skipping.")
            pbar.update(len(samples))
            continue

        contents = dataset.extract_document_contents(
            samples[0],
            load_contents=True,
            save_contents=True,
            clip_border=True,
        )

        page_embeds = run_page_embed(
            retriever=retriever,
            dataset=dataset,
            sample=samples[0],
            contents=contents,
            batch_size=batch_size,
            use_image_embeds=use_image_embeds,
            save_image_embeds=save_image_embeds,
            is_clip=is_clip
        )

        document_hypergraph = DocumentHypergraph()

        document_hypergraph.construct_page_similarity_graph(
            page_embeds=page_embeds,
            threshold=0.8,
            similarity_measure="cosine"
        )

        for sample in samples:

            document_hypergraph.clean_up_query_specific_hypergraph()

            queries, query_embeds = run_query_embed(
                retriever=retriever,
                vlm=vlm,
                sample=sample,
                question_key=dataset.question_key
            )

            top_page_indices_list, top_page_scores_list, col_score_dict, scores = run_late_interaction(
                page_embeds=page_embeds,
                query_embeds=query_embeds,
                top_k=top_k,
            )

            document_hypergraph._debug(
                dataset=dataset,
                sample=sample,
                queries=queries,
                scores=scores,
            )

            bandit = ThompsonSamplingBandit(alpha_prior=1, beta_prior=1)

            document_hypergraph.construct_query_specific_hypergraph(
                bandit=bandit,
                vlm=vlm,
                dataset=dataset,
                top_page_indices=top_page_indices_list,
                top_page_scores=top_page_scores_list,
                queries=queries,
                col_score_dict=col_score_dict,
                sample=sample,
            )

            top_page_indices, top_page_scores = bandit.mab_retrieval(
                document_hypergraph=document_hypergraph,
                vlm=vlm,
                dataset=dataset,
                sample=sample,
            )

            sample[retrieval_key] = top_page_indices
            sample[retrieval_key + "_score"] = top_page_scores

            pbar.update(1)

        path = dataset.dump_samples_file(all_samples, sample_name=sample_name, suffix=f"[retrieval_{retrival_name}]")
        print(f"[Info] [{CURRENT_FILE_PATH}] The {doc_id}'s Retrieval Results at {path}.")

    pbar.close()

if __name__ == "__main__":
    main()