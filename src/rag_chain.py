"""
RAG Chain — Retrieval-Augmented Generation pipeline
Uses LangChain to combine retrieval with Groq (Llama 3) generation.
"""

import os
import re
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

from config.settings import LLM_MODEL, DEFAULT_TOP_K
from src.utils import extract_response_text


RAG_SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant for an Obsidian vault.
You answer questions based ONLY on the provided context from the user's personal notes.

How to respond:
- Write in a natural, conversational tone like ChatGPT
- Use flowing paragraphs, not bullet points or lists
- Explain concepts clearly and thoroughly
- If the answer is not found in the context, say: "I couldn't find information about that in your notes. Could you rephrase your question or ask about something else in your vault?"
- At the very end of your response, mention which note(s) you used in a natural way, like "This information came from your [Note Name] notes." — but only if it flows naturally
- If the context contains relevant partial information, share what you can and note what's missing

Context from notes:
{context}

Question: {question}

Answer:"""

RAG_USER_PROMPT = """Based on your personal notes, here is the relevant information:

{context}

Your question: {question}"""


def _get_llm():
    """Create and return a Groq LLM (Llama 3)."""
    groq_key = os.environ.get("GROQ_API_KEY")

    return ChatGroq(
        groq_api_key=groq_key,
        model_name=LLM_MODEL,
        temperature=0.3,
        max_tokens=2048,
    )


def create_rag_chain(
    vector_store: Chroma,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple:
    """
    Create a RAG chain that retrieves relevant chunks and generates answers.
    Returns (rag_chain, retriever) tuple.
    """
    llm = _get_llm()

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_template(RAG_USER_PROMPT)

    # Build chain: retrieve → format → prompt → LLM → parse output
    def format_docs(docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", source)
            formatted.append(
                f"[{i}] Source: {source} ({title})\n{doc.page_content}"
            )
        return "\n\n".join(formatted)

    # Custom parser: extract text safely, then strip <thinking> tags
    def clean_response(raw) -> str:
        if isinstance(raw, str):
            text = raw
        else:
            text = extract_response_text(raw)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
        return text

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | clean_response
    )

    return rag_chain, retriever


def query_rag(
    rag_chain,
    question: str,
) -> Tuple[str, List[Document]]:
    """Query the RAG chain with a question."""
    answer = rag_chain.invoke(question)
    return answer, []


def query_rag_with_sources(
    vector_store: Chroma,
    question: str,
    chat_history: Optional[List] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[str, List[Document]]:
    """
    Complete RAG query that returns both the answer and source documents.
    """
    llm = _get_llm()

    # Retrieve relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    retrieved_docs = retriever.invoke(question)

    # Format context
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", source)
        context_parts.append(
            f"[{i}] Source: {source} ({title})\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Build prompt with optional chat history
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.title()}: {content}\n"
        history_text += "\n"

    system_msg = """You are a knowledgeable, friendly assistant for an Obsidian vault. Answer based ONLY on the provided context from the user's notes.

How to respond:
- Write in a natural, conversational tone like ChatGPT
- Use flowing paragraphs, not bullet points or lists
- Explain concepts clearly and thoroughly using natural language
- If the answer is not in the context, say "I couldn't find that in your notes."
- At the very end of your response, mention which note(s) you used in a natural way, like "This came from your [Note Name] notes." — but only if it flows naturally
- If context contains partial information, share what you can and note what's missing"""

    full_prompt = f"""{system_msg}

Context from notes:
{context}

{history_text}
Question: {question}

Answer:"""

    # Generate answer
    response = llm.invoke([HumanMessage(content=full_prompt)])

    # Extract clean text
    answer = extract_response_text(response)
    answer = re.sub(r'<thinking>.*?</thinking>', '', answer, flags=re.DOTALL).strip()

    return answer, retrieved_docs
