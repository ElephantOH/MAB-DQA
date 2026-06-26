import os
import sys
import hydra
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
CURRENT_FILE_PATH = os.path.abspath(__file__)

from src.prompt.prompt_mab import PROMPTS
from src.model.factory import ModelFactory
from src.until.late_interaction import late_interaction, get_top_page
from src.mab.document_hypergraph import DocumentHypergraph
from src.mab.thompson_sampling import ThompsonSamplingBandit

from typing import List


@dataclass
class Content:
    image: Image
    text: str


class LocalPDFDataset:

    def __init__(
        self,
        contents: List[Content],
        doc_name: str,
        dataset_name: str = "custom_pdf"
    ):
        self.contents = contents 
        self.doc_name = doc_name
        self.dataset_name = dataset_name

    @staticmethod
    def remove_image_border(image, tolerance=10, remove_inner=False, margin=1):

        if hasattr(image, 'samples') and hasattr(image, 'width') and hasattr(image, 'height'):
            image_mode = "RGBA" if image.alpha else "RGB"
            pil_image = Image.frombytes(image_mode, (image.width, image.height), image.samples)
        else:
            pil_image = image.convert("RGB")

        image_np_array = np.array(pil_image)
        if image_np_array.ndim == 2:
            image_np_array = np.stack([image_np_array]*3, axis=-1)
        
        image_height, image_width, color_channels = image_np_array.shape

        corner_pixels = [
            image_np_array[0, 0],
            image_np_array[0, image_width - 1],
            image_np_array[image_height - 1, 0],
            image_np_array[image_height - 1, image_width - 1]
        ]
        background_color = np.mean(corner_pixels, axis=0)
        color_difference = np.abs(image_np_array - background_color)
        is_background_pixel = np.all(color_difference <= tolerance, axis=2)

        if remove_inner:
            if np.all(is_background_pixel):
                return pil_image

            non_background_rows = np.where(~np.all(is_background_pixel, axis=1))[0]
            non_background_cols = np.where(~np.all(is_background_pixel, axis=0))[0]
            if len(non_background_rows) == 0 or len(non_background_cols) == 0:
                return pil_image.crop((0, 0, 1, 1))

            cropped_image_array = image_np_array[non_background_rows, :, :]
            cropped_image_array = cropped_image_array[:, non_background_cols, :]
            cropped_pil_image = Image.fromarray(cropped_image_array)

            if margin > 0:
                padded_width = cropped_image_array.shape[1] + 2 * margin
                padded_height = cropped_image_array.shape[0] + 2 * margin
                padded_pil_image = Image.new(
                    pil_image.mode, 
                    (padded_width, padded_height), 
                    tuple(background_color.astype(int))
                )
                padded_pil_image.paste(cropped_pil_image, (margin, margin))
                return padded_pil_image
            return cropped_pil_image

        is_full_background_row = np.all(is_background_pixel, axis=1)
        top_bound = np.argmax(~is_full_background_row)
        bottom_bound = image_height - 1 - np.argmax(~is_full_background_row[::-1])

        is_full_background_col = np.all(is_background_pixel, axis=0)
        left_bound = np.argmax(~is_full_background_col)
        right_bound = image_width - 1 - np.argmax(~is_full_background_col[::-1])

        if top_bound >= bottom_bound or left_bound >= right_bound:
            return pil_image.crop((0, 0, 1, 1))

        top_bound = max(0, top_bound - margin)
        bottom_bound = min(image_height, bottom_bound + margin + 1)
        left_bound = max(0, left_bound - margin)
        right_bound = min(image_width, right_bound + margin + 1)

        return pil_image.crop((left_bound, top_bound, right_bound, bottom_bound))

    def extract_page_contents(
        self,
        sample: dict,
        page: int,
        load_contents: bool = True,
        save_contents: bool = False,
        clip_border: bool = True,
    ) -> List[Content]:

        if page < 0 or page >= len(self.contents):
            raise IndexError(f"Page {page} out of range, total pages: {len(self.contents)}")
        
        original_content = self.contents[page]
        
        if clip_border:
            cropped_image = self.remove_image_border(original_content.image)

            return [Content(image=cropped_image, text=original_content.text)]
        
        return [original_content]


class MABRetrieverTools:
    def __init__(
            self,
            retriever: str = "colpali/colpali-1.3",
            vlm: str = "qwen/qwen25vl-7b",
            top_k: int = 10,
            cache_embeds: bool = True,
            clip_border: bool = True
        ):
        
        print("[Initialization] Loading models... It may take time for the first load, please wait patiently")
        self.retriever_type = retriever
        self.vlm_type = vlm
        self.top_k = top_k
        self.cache_embeds = cache_embeds
        self.clip_border = clip_border
        self.batch_size = 16
        self.is_clip = True
        self.question_key = "question"
      
        self.retrieval_device = f"cuda:{0}"
        self.vlm_device = f"auto"
        self.retrival_name = "mab"
        self.retrieval_key = f"retrieval[{self.retrival_name}]"
  
        current_dir = Path(__file__).parent.resolve()
        config_dir = (current_dir / "../../../configs").resolve()

        with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
            self.config = hydra.compose(config_name="config")

        self._init_models()

        self.embed_cache = {}
        print("[Initialization] Models loaded successfully!")

    def _init_models(self):
        self.retriever = ModelFactory.create(
            model_type=self.retriever_type,
            config=self.config,
            device_map=self.retrieval_device,
            dtype="bfloat16",
        )
        self.vlm = ModelFactory.create(
            model_type=self.vlm_type,
            config=self.config,
            device_map=self.vlm_device,
            dtype="bfloat16",
        )

    def _extract_pdf_contents(self, pdf_path: str, dpi: int = 180):
        import pymupdf
        contents = []
        doc_name = Path(pdf_path).stem
        
        with pymupdf.open(pdf_path) as pdf:
            for index, page in enumerate(pdf):
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                text = page.get_text("text")
                contents.append(Content(image=image, text=text))
        
        return contents, doc_name

    def _run_page_embed(self, contents, doc_name):

        cache_key = f"{doc_name}_clip_{self.clip_border}"
        
        if self.cache_embeds and cache_key in self.embed_cache:
            print(f"[Cache] Hit {doc_name} page embeddings (clip_border={self.clip_border}), skip regeneration")
            return self.embed_cache[cache_key]

        images = []
        for content in contents:
            if self.clip_border:
                img = LocalPDFDataset.remove_image_border(content.image)
            else:
                img = content.image
            images.append(img)

        page_embeds = self.retriever.encode_images(images=images, batch_size=self.batch_size)
        
        if self.cache_embeds:
            self.embed_cache[cache_key] = page_embeds
        
        return page_embeds

    def _run_query_embed(self, question: str, sample: dict):
        initial_query = question
        sample["initial_query"] = initial_query

        basis_queries, _ = self.vlm.predict(
            question=PROMPTS["query"],
            texts=[initial_query],
            images=None,
            max_new_tokens=60,
        )
        sample["basis_queries"] = basis_queries

        queries = [item.strip() for item in basis_queries.split(',')]
        queries.extend([basis_queries, initial_query])
        query_embeds = self.retriever.encode_queries(queries)
        
        return queries, query_embeds

    def _mab_retrieval(self, page_embeds, query_embeds, queries, dataset, sample):
        scores, col_score_dict = late_interaction(page_embeds, query_embeds)
        top_page = get_top_page(scores, self.top_k)
        
        top_page_indices_list = []
        top_page_scores_list = []
        for i in range(scores.shape[0]):
            indices = top_page.indices.tolist()[i] if top_page is not None else []
            scores_i = top_page.values.tolist()[i] if top_page is not None else []
            top_page_indices_list.append(indices)
            top_page_scores_list.append(scores_i)

        document_hypergraph = DocumentHypergraph()
        document_hypergraph.construct_page_similarity_graph(
            page_embeds=page_embeds, threshold=0.8, similarity_measure="cosine"
        )
        document_hypergraph.clean_up_query_specific_hypergraph()
        
        bandit = ThompsonSamplingBandit(alpha_prior=1, beta_prior=1)
        document_hypergraph.construct_query_specific_hypergraph(
            bandit=bandit,
            vlm=self.vlm,
            dataset=dataset,
            top_page_indices=top_page_indices_list,
            top_page_scores=top_page_scores_list,
            queries=queries,
            col_score_dict=col_score_dict,
            sample=sample
        )

        top_page_indices, top_page_scores = bandit.mab_retrieval(
            document_hypergraph=document_hypergraph,
            vlm=self.vlm,
            dataset=dataset, 
            sample=sample
        )
        
        return top_page_indices, top_page_scores
    
    def retrieve(self, pdf_path: str, question: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
        
        print(f"\n[Processing] PDF: {pdf_path}")
        print(f"[Question]: {question}")
        print(f"[Config] clip_border: {self.clip_border}")

        contents, doc_name = self._extract_pdf_contents(pdf_path)
        print(f"[Info] Total pages in PDF: {len(contents)}")

        dataset = LocalPDFDataset(
            contents=contents,
            doc_name=doc_name,
            dataset_name="custom_pdf"
        )

        sample = {
            self.question_key: question,
            "doc_id": f"{doc_name}.pdf",
        }

        page_embeds = self._run_page_embed(contents, doc_name)

        queries, query_embeds = self._run_query_embed(question, sample)

        top_pages, top_scores = self._mab_retrieval(
            page_embeds=page_embeds,
            query_embeds=query_embeds,
            queries=queries,
            dataset=dataset,
            sample=sample
        )

        print(f"[Result] Top {self.top_k} relevant pages: {top_pages}")
        print(f"[Scores]: {[round(s,4) for s in top_scores]}")
        
        return top_pages, top_scores


# if __name__ == "__main__":
#     retriever = MABRetrieverTools(top_k=10, clip_border=True)
#
#     while True:
#         print("\n" + "-"*50)
#         pdf_path = "examples/example.pdf"
#         question = "what is total current assets in FY2023 for Bestbuy? Answer in million."
#         
#         try:
#             pages, scores = retriever.retrieve(pdf_path, question)
#         except Exception as e:
#             print(f"Error: {e}")
