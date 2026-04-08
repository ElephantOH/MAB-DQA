import os
import sys
import hydra
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
CURRENT_FILE_PATH = os.path.abspath(__file__)

from src.prompt.prompt_mab import PROMPTS
from src.model.factory import ModelFactory
from src.until.late_interaction import late_interaction, get_top_page
from src.mab.document_hypergraph import DocumentHypergraph
from src.mab.thompson_sampling import ThompsonSamplingBandit

@dataclass
class Content:
    image: Image
    text: str

class MABRetrieverTools:
    def __init__(
            self,
            retriever: str = "colpali/colpali-1.3",
            vlm: str = "qwen/qwen25vl-7b",
            top_k: int = 10,
            cache_embeds: bool = True
        ):
        
        print("[Initialization] Loading models... It may take time for the first load, please wait patiently")
        self.retriever_type = retriever
        self.vlm_type = vlm
        self.top_k = top_k
        self.cache_embeds = cache_embeds
        self.batch_size = 16
        self.is_clip = True
        self.question_key = "question"
      
        self.retrieval_device = f"cuda:{0}"
        self.vlm_device = f"auto"
        self.retrival_name = "mab"
        self.retrieval_key = f"retrieval[{self.retrival_name}]"
  
        with hydra.initialize_config_dir(config_dir="../configs", version_base=None):
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
                image = page.get_pixmap(dpi=dpi)
                if hasattr(image, 'samples'):
                    mode = "RGB" if not image.alpha else "RGBA"
                    image = Image.frombytes(mode, (image.width, image.height), image.samples)
                text = page.get_text("text")
                contents.append(Content(image=image, text=text))
        
        return contents, doc_name

    def _run_page_embed(self, contents, doc_name):

        sample = {"doc_id": f"{doc_name}.pdf"}
        
        if self.cache_embeds and doc_name in self.embed_cache:
            print(f"[Cache] Hit {doc_name} page embeddings, skip regeneration")
            return self.embed_cache[doc_name]

        images = [content.image for content in contents]
        page_embeds = self.retriever.encode_images(images=images, batch_size=self.batch_size)
        
        if self.cache_embeds:
            self.embed_cache[doc_name] = page_embeds
        
        return page_embeds

    def _run_query_embed(self, question: str):

        sample = {self.question_key: question}
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

    def _mab_retrieval(self, page_embeds, query_embeds, queries):

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
            bandit=bandit, vlm=self.vlm, dataset=None,
            top_page_indices=top_page_indices_list,
            top_page_scores=top_page_scores_list,
            queries=queries, col_score_dict=col_score_dict, sample={}
        )

        top_page_indices, top_page_scores = bandit.mab_retrieval(
            document_hypergraph=document_hypergraph, vlm=self.vlm, dataset=None, sample={}
        )
        
        return top_page_indices, top_page_scores

    def retrieve(self, pdf_path: str, question: str):

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
        
        print(f"\n[Processing] PDF: {pdf_path}")
        print(f"[Question]: {question}")
        
        contents, doc_name = self._extract_pdf_contents(pdf_path)
        print(f"[Info] Total pages in PDF: {len(contents)}")

        page_embeds = self._run_page_embed(contents, doc_name)

        queries, query_embeds = self._run_query_embed(question)

        top_pages, top_scores = self._mab_retrieval(page_embeds, query_embeds, queries)

        print(f"[Result] Top {self.top_k} relevant pages: {top_pages}")
        print(f"[Scores]: {[round(s,4) for s in top_scores]}")
        
        return top_pages, top_scores


# if __name__ == "__main__":

#     retriever = MABRetrieverTools(top_k=10)

#     while True:
#         print("\n" + "-"*50)
#         pdf_path = "examples/example.pdf"
#         question = "what is total current assets in FY2023 for Bestbuy? Answer in million."
        
#         try:
#             pages, scores = retriever.retrieve(pdf_path, question)
#         except Exception as e:
#             print(f"Error: {e}")
