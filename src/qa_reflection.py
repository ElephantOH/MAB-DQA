import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
CURRENT_FILE_PATH = os.path.abspath(__file__)

import torch
import hydra
from tqdm import tqdm
from time import time
from concurrent import futures
from src.prompt.prompt_mab import prompt_manager
from src.data.factory import DatasetFactory
from src.model.factory import ModelFactory


# %%

def avg(numbers):
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

#%%


def qa_agent_predict(qa_agent, sample, question, texts, images):
 
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        prompt = prompt_manager["simple_top4"](question)
        image_start_time = time.time()
        image_future = executor.submit(
            qa_agent.predict,
            prompt,
            texts=None,
            images=images[:4],
            with_sys_prompt=False,
        )

        image_response, _ = image_future.result()
        image_time = time.time() - image_start_time

    print("\n" + "=" * 30)
    print(f"STAGE 1: INITIAL RESPONSE (Time Cost: {image_time:.2f} sec):")
    print("=" * 30)
    print((image_response))

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        validation_prompt = prompt_manager["answer_validation"](question, image_response)
        validation_start_time = time.time()
        validation_future = executor.submit(
            qa_agent.predict,
            validation_prompt,
            texts=None,
            images=images[:4],
            with_sys_prompt=False,
        )

        validation_response, _ = validation_future.result()
        validation_time = time.time() - validation_start_time

    validation_result = validation_response.strip().lower()

    print("\n" + "=" * 30)
    print(f"STAGE 2: ANSWER VALIDATION (Time Cost: {validation_time:.2f} sec):")
    print("=" * 30)
    print(f"Validation result: {validation_result}")

    if "yes" in validation_result:
        validation_result = 1
    elif "no" in validation_result:
        validation_result = 0

    if validation_result == 0:
        basis_queries = sample.get("basis_queries")
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            basis_prompt = prompt_manager["basis_analysis"](question, basis_queries)
            basis_start_time = time.time()
            basis_future = executor.submit(
                qa_agent.predict,
                basis_prompt,
                texts=None,
                images=images[:4],
                with_sys_prompt=False,
            )

            basis_response, _ = basis_future.result()
            basis_time = time.time() - basis_start_time

        print("\n" + "=" * 30)
        print(f"STAGE 3.1: BASIS ANALYSIS (Time Cost: {basis_time:.2f} sec)")
        print("=" * 30)
        print((basis_response))

        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            summary_prompt = prompt_manager["text_summary"](question, basis_response)
            summary_start_time = time.time()
            summary_future = executor.submit(
                qa_agent.predict,
                summary_prompt,
                texts=None,
                images=images[:4],
                with_sys_prompt=False,
            )

            summary_response, _ = summary_future.result()
            summary_time = time.time() - summary_start_time

        print("\n" + "=" * 30)
        print(f"STAGE 3.2: TEXT SUMMARY (Time Cost: {summary_time:.2f} sec)")
        print("=" * 30)
        print((summary_response))

        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            refined_prompt = prompt_manager["refined_qa"](question, image_response, summary_response, basis_response)
            refined_start_time = time.time()
            refined_future = executor.submit(
                qa_agent.predict,
                refined_prompt,
                texts=None,
                images=images[:4],
                with_sys_prompt=False,
            )

            final_response, _ = refined_future.result()
            refined_time = time.time() - refined_start_time

        print("\n" + "=" * 30)
        print(f"STAGE 3.3: REFINED ANSWER (Time Cost: {refined_time:.2f} sec):")
        print("=" * 30)
        print((final_response))

        image_response = final_response

    return image_response, validation_result

# %%

def parse_evidence_pages(evidence_str, shift=-1):
    if isinstance(evidence_str, list):
        pages = [int(page) for page in evidence_str]
    else:
        evidence_str = evidence_str.strip('[]')
        if not evidence_str:
            return []
        pages = [int(page.strip()) for page in evidence_str.split(',')]

    adjusted_pages = []
    for page in pages:
        shifted_page = page + shift
        if shifted_page < 0:
            shifted_page = 0
        adjusted_pages.append(shifted_page)

    return adjusted_pages


def analyze_results(cfg, qa_name, sample, avg_cor_idx, cor_idx):
    try:
        cor_flag = "✓"
        if cfg.dataset.name == "MMLongBench":
            shift = -1
        else:
            shift = 0
        if cor_idx < 0.5:
            cor_flag = "×"

        print("\n" + "*" * 60)
        if 'evidence_pages' in sample:
            print(f"Evidence Pages: {str(parse_evidence_pages(sample['evidence_pages'], shift))}")
        if 'evidence_sources' in sample:
            print(f"Evidence Sources: {str(sample['evidence_sources'])}")
        if 'text-top-10-question' in sample:
            print(f"Text RAG: {str(sample['text-top-10-question'])}")
        if 'image-top-10-question' in sample:
            print(f"Image RAG: {str(sample['image-top-10-question'])}")
    except Exception as e:
        print(f"Error in evidence: {str(e)}")

    try:
        print("-" * 60)
        print(f"Question: {(str(sample[cfg.dataset.question_key]))}")
        if f"ans_{qa_name}" in sample:
            print(f"[ {cor_flag} ] Agent Answer: {(str(sample[f'ans_{qa_name}']))}")
        print("-" * 60)
        print(f"Reference Answer: {(str(sample[cfg.dataset.gt_key]))}")
        print(f"[{cfg.dataset.name}] Acc: {avg_cor_idx:.3f}")
        print("*" * 60)
    except Exception as e:
        print(f"Error in analyze_results: {str(e)}")


# %%
@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def main(config):
    wait_hours = 0

    sample_name = "samples"
    retrival_name = "mab"
    qa_name = "val"

    qa_agent_device = f"balanced_low_0"
    resume = False
    question_key = "question"
    retrieval_key = f"retrieval[{retrival_name}]"
    answer_key = f"answer[{qa_name}]"

    wait_seconds = wait_hours * 3600
    for remaining in tqdm(range(wait_seconds, 0, -1), desc="Waiting...", unit="s"):
        time.sleep(1)
   
    qa_agent = ModelFactory.create(
        model_type="qwen/qwen25vl-7b",
        config=config,
        device_map=qa_agent_device,
        dtype="bfloat16",
    )

    eval_agent = ModelFactory.create(
        model_type="openai/gpt-4o",
        config=config,
    )

    dataset = DatasetFactory.create(
        mission_key="close_domin_dqa",
        dataset_name="mmlb",
        config=config,
        question_key=question_key,
        retrieval_key=retrieval_key,
        answer_key=answer_key,
    )

    if resume:
        samples = dataset.load_samples_file(sample_name=sample_name, suffix=f"[retrieval_{retrival_name}][qa_{qa_name}]")
    else:
        samples = dataset.load_samples_file(sample_name=sample_name, suffix=f"[retrieval_{retrival_name}]")

    print("*" * 60)
    print(f"[Info] [{CURRENT_FILE_PATH}] Config:")
    print(config)
    print("*" * 60)

    total_samples = len(samples)

    pbar = tqdm(total=total_samples, desc="Processing samples")
    doucument_samples = dataset.group_samples_by_doc_id(samples)

    for doc_id, sample_list in doucument_samples.items():

        all_processed = True
        for sample in sample_list:
            if answer_key not in sample:
                all_processed = False
                break

        if all_processed:
            print(f"[Info] [{CURRENT_FILE_PATH}] Document {doc_id} already Processed, Skipping.")
            pbar.update(len(sample_list))
            continue

        for sample in sample_list:
            question, texts, images, text_pages, image_pages = dataset.get_retrieval_results(sample, top_k=10)
            print(f"[Info] Question: {(question)} \n")
            try:
                final_ans, self_val = qa_agent_predict(qa_agent, sample, question, texts, images)
            except RuntimeError as e:

                if "CUDA out of memory" in str(e) or "out of memory" in str(e):

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        qa_agent.clean_messages()
                final_ans = f"[Error] Model prediction failed. {e}."
                self_val = 0
                print(f"[Error]: {e}")

            sample[f"ans_{qa_name}"] = final_ans
            sample[f"self_val"] = self_val

            torch.cuda.empty_cache()
            qa_agent.clean_messages()

            path = dataset.dump_data(samples, sample_name=sample_name, suffix=f"[retrieval_{retrival_name}][qa_{qa_name}]")
            print(f"[Info] [{CURRENT_FILE_PATH}] The {doc_id}'s Retrieval Results at {path}.")
            pbar.update(1)


if __name__ == "__main__":
    main()