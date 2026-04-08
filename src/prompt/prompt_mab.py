import os
from typing import Callable, Dict
from dataclasses import dataclass

CURRENT_FILE_PATH = os.path.abspath(__file__)

# ===================== 1. Extract Public Constants: Manage all repeated text uniformly, modify in one place and take effect globally =====================
# ---------------- Retrieval Scoring & Query Processing ----------------
RETRIEVAL_GOAL = \
"# GOAL #\nYou are an Retrieval Expert, and your task is to evaluate how relevant the input document page is to the given query."

RATING_NORMAL = """ \
Rate the relevance on a scale of 1 to 5, where:
- 5: Highly relevant - contains complete information needed to answer the query
- 4: Very relevant - contains most of the information needed
- 3: Moderately relevant - contains some useful information
- 2: Slightly relevant - has minor connection to the query
- 1: Irrelevant - contains no information related to the query
"""

RATING_DETAILED = """ \
Rate the relevance on a scale of 1 to 5, where:
- 5: Highly relevant - contains COMPLETE information to fully answer the query (be cautious with this rating)
- 4: Very relevant - contains most information needed but may lack some details (be cautious with this rating)
- 3: Moderately relevant - contains some useful information but significant gaps remain
- 2: Slightly relevant - has minor connection to the query
- 1: Irrelevant - contains no information related to the query
"""

INSTRUCTION_NORMAL = """\
# INSTRUCTION #\nPlease first read the given query, think about what knowledge is required to answer that query, and then carefully go through the document snapshot for judgment.
"""

INSTRUCTION_DETAILED = """ \
# INSTRUCTION #\nPlease first read the given query, think about what specific information is required to answer that query comprehensively, and then carefully examine the document snapshot.
IMPORTANT: Before giving a score of 4 or 5, verify that the page actually contains the specific facts needed to answer the query, not just related information."""

INSTRUCTION_CONDITIONAL = "# INSTRUCTION #\nBased on previous retrieval system judgment, we believe that this document snapshot is at least '{}' relevant. "

OUTPUT_NORMAL = "\nPlease generate just a single number (1-5) representing your relevance judgment. Your answer should be a single number without any extra contents. "
OUTPUT_DETAILED = "\nThink step by step about the relevance, then provide just a single number (1-5) representing your judgment."

QUERY_PROMPT = """ \
As an AI agent specialized in document retrieval query processing, your primary task is to handle each query by first ignoring any irrelevant information (such as output format requests or non-retrieval instructions).
Then, extract meaningful entities and key phrases that capture the core intent of the query.
Finally, output the result as a comma-separated list of key phrases, for example: "key_phrase1, key_phrase2, ...". Ensure clarity and conciseness throughout.
"""

# ---------------- Q&A Related ----------------
SIMPLE_TOP4_PROMPT = """
Using the provided four document screenshots, answer this question: "{}"
Requirements:
- Reply must be extremely concise (as short as possible)
- Use only information visible in the screenshots
- If the answer cannot be clearly found, respond exactly: "Not answerable"
Answer:
"""

ANSWER_VALIDATION_PROMPT = """
You will be given a question and a corresponding answer. Your task is to determine whether the answer addresses the question, regardless of whether the answer is correct or not.
Focus only on whether the answer responds to the question and covers the necessary points (i.e., no essential content is missing).
If no answer is provided, consider it as not answering.

Question: {}
Answer: {}

Did the answer address the question? (yes/no)
"""

BASIS_ANALYSIS_PROMPT = """
Analyze the following question and identify the core concepts and relationships that need to be understood to answer it properly.

Question: "{}"
{}

Requirements:
- Break down the question into fundamental components
- Identify what specific information is needed to answer each component
- Note any implicit relationships or assumptions in the question
- Be concise but thorough in your analysis

Analysis:
"""

TEXT_SUMMARY_PROMPT = """
Analyze the provided document screenshots and summarize all information relevant to the following question: "{}"

{}

Requirements:
- Extract and summarize information that is directly related to the question
- Be comprehensive but concise
- Include key facts, figures, and details that could help answer the question
- Organize information logically based on the question's components

Summary of relevant information:
"""

REFINED_QA_PROMPT = """
Based on the following context, provide a better answer to the question through careful reasoning.

Question: {}
Initial incomplete answer: {}
Relevant information summary: {}
{}

CRITICAL THINKING REQUIREMENTS:
1. First, analyze what the question is REALLY asking for
2. Compare the initial answer with the available information
3. Identify gaps or inaccuracies in the initial answer
4. Synthesize information from the summary to fill these gaps
5. Formulate a coherent response that directly addresses the question

DO NOT simply copy phrases from the summary. Instead, use the information to construct a thoughtful answer.

If the summary indicates no relevant information, respond: "Not answerable"

Reasoning process:
- [Analyze the question requirements]
- [Compare initial answer with evidence]
- [Identify what needs to be improved]
- [Synthesize the improved answer]

Improved answer:
"""

# ===================== 2. Class Encapsulation: Prompt Manager =====================
@dataclass(frozen=True)
class PromptManager:
    """Unified Prompt Manager for retrieval tasks, provides type-safe Prompt generation interface with parameter validation"""

    # ---------------- Basic Query Processing ----------------
    def get_query(self) -> str:
        """Get the prompt for extracting query keywords"""
        return QUERY_PROMPT

    # ---------------- Normal Retrieval Scoring ----------------
    def get_retrieval(self, query: str) -> str:
        """
        Generate normal retrieval scoring prompt
        :param query: User query statement
        """
        self._validate_query(query)
        return f"""{RETRIEVAL_GOAL}{RATING_NORMAL}{INSTRUCTION_NORMAL}# QUERY #{query}{OUTPUT_NORMAL}"""

    # ---------------- Conditional Retrieval Scoring with Prior Score ----------------
    def get_conditional_retrieval(self, priori: str, query: str) -> str:
        """
        Generate conditional retrieval prompt with prior score
        :param priori: Prior relevance level
        :param query: User query statement
        """
        self._validate_query(query)
        self._validate_priori(priori)
        instruction = INSTRUCTION_CONDITIONAL.format(priori) + INSTRUCTION_NORMAL.split("# INSTRUCTION #\n")[1]
        return f"""{RETRIEVAL_GOAL}{RATING_NORMAL}{instruction}# QUERY #{query}{OUTPUT_NORMAL}"""

    # ---------------- Detailed Retrieval Scoring ----------------
    def get_retrieval_detailed(self, query: str) -> str:
        self._validate_query(query)
        return f"""{RETRIEVAL_GOAL}{RATING_DETAILED}{INSTRUCTION_DETAILED}# QUERY #{query}{OUTPUT_DETAILED}"""

    # ---------------- Detailed Conditional Retrieval Scoring with Prior Score ----------------
    def get_conditional_retrieval_detailed(self, priori: str, query: str) -> str:
        self._validate_query(query)
        self._validate_priori(priori)
        instruction = INSTRUCTION_CONDITIONAL.format(priori) + INSTRUCTION_DETAILED.split("# INSTRUCTION #\n")[1]
        return f"""{RETRIEVAL_GOAL}{RATING_DETAILED}{instruction}# QUERY #{query}{OUTPUT_DETAILED}"""

    # ---------------- Simple Top4 Q&A ----------------
    def get_simple_top4(self, question: str) -> str:
        """
        Generate prompt for simple Q&A with top 4 document screenshots
        :param question: User question
        """
        self._validate_query(question)
        return SIMPLE_TOP4_PROMPT.format(question)

    # ---------------- Answer Validation ----------------
    def get_answer_validation(self, question: str, answer: str) -> str:
        """
        Generate prompt for validating if answer addresses the question
        :param question: User question
        :param answer: Answer to validate
        """
        self._validate_query(question)
        self._validate_answer(answer)
        return ANSWER_VALIDATION_PROMPT.format(question, answer)

    # ---------------- Question Basis Analysis ----------------
    def get_basis_analysis(self, question: str, basis_queries: str | None = None) -> str:
        """
        Generate prompt for analyzing question core concepts
        :param question: User question
        :param basis_queries: Optional key aspects to focus on
        """
        self._validate_query(question)
        extra = f"Key aspects to focus on: {basis_queries}" if basis_queries else "Identify the key concepts and relationships in this question."
        return BASIS_ANALYSIS_PROMPT.format(question, extra)

    # ---------------- Text Relevance Summary ----------------
    def get_text_summary(self, question: str, basis_analysis: str | None = None) -> str:
        """
        Generate prompt for summarizing relevant document information
        :param question: User question
        :param basis_analysis: Optional question analysis basis
        """
        self._validate_query(question)
        extra = basis_analysis if basis_analysis else ""
        return TEXT_SUMMARY_PROMPT.format(question, extra)

    # ---------------- Refined Q&A Reasoning ----------------
    def get_refined_qa(self, question: str, initial_answer: str, summary: str, basis_analysis: str | None = None) -> str:
        """
        Generate prompt for refining answer with reasoning
        :param question: User question
        :param initial_answer: Initial incomplete answer
        :param summary: Relevant information summary
        :param basis_analysis: Optional question analysis basis
        """
        self._validate_query(question)
        self._validate_answer(initial_answer)
        self._validate_summary(summary)
        extra = basis_analysis if basis_analysis else ""
        return REFINED_QA_PROMPT.format(question, initial_answer, summary, extra)

    # ===================== Parameter Validation: Prevent null/illegal parameters =====================
    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not query.strip():
            raise ValueError(f"[Error] [{CURRENT_FILE_PATH}] Query/Question cannot be Empty!")

    @staticmethod
    def _validate_priori(priori: str) -> None:
        if not priori or not priori.strip():
            raise ValueError(f"[Error] [{CURRENT_FILE_PATH}] Prior Level cannot be Empty!")

    @staticmethod
    def _validate_answer(answer: str) -> None:
        if not answer or not answer.strip():
            raise ValueError(f"[Error] [{CURRENT_FILE_PATH}] Answer cannot be Empty!")

    @staticmethod
    def _validate_summary(summary: str) -> None:
        if not summary or not summary.strip():
            raise ValueError(f"[Error] [{CURRENT_FILE_PATH}] Summary cannot be Empty!")


# Initialize singleton instance
prompt_manager = PromptManager()

PROMPTS: Dict[str, Callable | str] = {
    # Retrieval Prompts
    "query": prompt_manager.get_query(),
    "retrieval": prompt_manager.get_retrieval,
    "conditional_retrieval": prompt_manager.get_conditional_retrieval,
    "retrieval_detailed": prompt_manager.get_retrieval_detailed,
    "conditional_retrieval_detailed": prompt_manager.get_conditional_retrieval_detailed,
    # Q&A Prompts
    "simple_top4": prompt_manager.get_simple_top4,
    "answer_validation": prompt_manager.get_answer_validation,
    "basis_analysis": prompt_manager.get_basis_analysis,
    "text_summary": prompt_manager.get_text_summary,
    "refined_qa": prompt_manager.get_refined_qa,
}
