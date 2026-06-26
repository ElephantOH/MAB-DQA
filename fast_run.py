from src.tools.close_domain_dqa.mab_retrieval_tools import MABRetrieverTools

if __name__ == "__main__":
    retriever = MABRetrieverTools(
        retriever="colpali/colpali-1.3",
        vlm="qwen/qwen25vl-7b",
        top_k=10,
    )
    
    pdf_path = "examples/example.pdf"
    question = "what is total current assets in FY2023 for Bestbuy? Answer in million."
    
    pages, scores = retriever.retrieve(pdf_path, question)
