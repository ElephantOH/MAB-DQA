from src.tools.close_domain_dqa.mab_retrieval_tools import MABRetrieverTools

if __name__ == "__main__":
    retriever = MABRetrieverTools(
        retriever_type="colpali/colpali-1.3",
        vlm_type="qwen/qwen25vl-7b",
        top_k=10,
    )
    while True:
        print("\n" + "-"*50)
        pdf_path = "examples/example.pdf"
        question = "what is total current assets in FY2023 for Bestbuy? Answer in million."
        try:
            pages, scores = retriever.retrieve(pdf_path, question)
        except Exception as e:
            print(f"Error: {e}")