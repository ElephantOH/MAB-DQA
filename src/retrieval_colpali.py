import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
CURRENT_FILE_PATH = os.path.abspath(__file__)

import hydra
import torch
import argparse
from tqdm import tqdm
from src.model.factory import ModelFactory
from src.data.factory import DatasetFactory
from src.until.late_interaction import late_interaction, get_top_page

def parse_args():

    parser = argparse.ArgumentParser(description="Colpali Retrieval Pipeline")
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

#%%
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
        sample,
        question_key="question",

    ):
    assert question_key in sample, f"[Error] [{CURRENT_FILE_PATH}] Question Key '{question_key}' not Found in Sample."
    queries = [sample[question_key]]
    query_embeds = retriever.encode_queries(queries)
    return query_embeds

def run_late_interaction(
        page_embeds,
        query_embeds,
        top_k=10,
    ):
    scores, _ = late_interaction(page_embeds, query_embeds)
    top_page = get_top_page(scores, top_k)

    top_page_indices = top_page.indices.tolist()[0] if top_page is not None else []
    top_page_scores = top_page.values.tolist()[0] if top_page is not None else []

    return top_page_indices, top_page_scores


#%%
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config):
    args = parse_args()
    retrival_name = "colpali"
    sample_name = "samples"
    retrieval_device = f"cuda:{0}"
    resume = False

    batch_size = 16
    use_image_embeds = True
    save_image_embeds = True
    is_clip = False
    question_key = "question"
    retrieval_key = f"retrieval[{retrival_name}]"
    top_k = 10

    retriever = ModelFactory.create(
        model_type="colpali/colpali-1.3",
        config=config,
        device_map=retrieval_device,
        dtype="bfloat16"
    )

    dataset = DatasetFactory.create(
        mission_key="close_domain_dqa",
        dataset_type="mmlb",
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

    document_samples = dataset.group_samples_by_doc_id(all_samples)

    for doc_id, samples in document_samples.items():

        all_processed = True
        for sample in samples:
            if (dataset.retrieval_key not in sample):
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
            clip_border=False,
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

        for sample in samples:

            query_embeds = run_query_embed(
                retriever=retriever,
                sample=sample,
                question_key=dataset.question_key
            )

            top_page_indices, top_page_scores = run_late_interaction(
                page_embeds=page_embeds,
                query_embeds=query_embeds,
                top_k=top_k,
            )

            sample[dataset.retrieval_key] = top_page_indices
            sample[dataset.retrieval_key + "_score"] = top_page_scores

            pbar.update(1)

        path = dataset.dump_samples_file(all_samples, sample_name=sample_name, suffix=f"[retrieval_{retrival_name}]")
        print(f"[Info] [{CURRENT_FILE_PATH}] The {doc_id}'s Retrieval Results at {path}.")

    pbar.close()


if __name__ == "__main__":
    main()