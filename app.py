import io
import os
import re
from typing import List, Dict, Any

import faiss
import fitz
import numpy as np
import requests
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Sindh Shops & Commercial Establishment Act — RAG Assistant"

# The user-provided Google Drive direct-download URL.
PDF_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1O_CjQSmShfJovVV9sU6NQ_mqZPFll17Q"
    "&export=download&authuser=0"
)

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# PAGE / UI
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Sindh Shops & Commercial Establishment Act — RAG Assistant")
st.caption(
    "Source-grounded Q&A over the supplied Sindh Shops and Commercial Establishment "
    "Act, 2015 (Sindh Act No. XII of 2016) PDF, including the amendments contained "
    "in the supplied document."
)

with st.sidebar:
    st.header("⚙️ Answer Controls")

    technicality = st.select_slider(
        "Technicality",
        options=["Simple", "Standard", "Technical", "Legal/Expert"],
        value="Standard",
        help="Controls the legal vocabulary and depth of explanation.",
    )

    response_size = st.select_slider(
        "Response size",
        options=["Short", "Medium", "Long", "Very long"],
        value="Medium",
    )

    language = st.selectbox(
        "Answer language",
        ["English", "Urdu", "Roman Urdu"],
        index=0,
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=2,
        max_value=8,
        value=5,
        help="More passages can improve coverage but may add noise.",
    )

    temperature = st.slider(
        "Creativity / strictness",
        min_value=0.0,
        max_value=0.8,
        value=0.1,
        step=0.1,
        help="For legal Q&A, a low value is recommended.",
    )

    reasoning = st.select_slider(
        "Reasoning effort",
        options=["low", "medium", "high"],
        value="medium",
    )

    show_sources = st.checkbox("Show retrieved source passages", value=True)
    strict_mode = st.checkbox(
        "Strict source mode",
        value=True,
        help="When enabled, the model must say when the supplied document does not "
             "contain enough information instead of filling gaps from general knowledge.",
    )

    st.divider()
    st.subheader("🔐 Groq API key")

    # Streamlit Cloud: put GROQ_API_KEY in App → Settings → Secrets.
    # Colab/local: the optional field makes testing easier without editing this file.
    secret_key = ""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    manual_key = st.text_input(
        "Optional API key",
        type="password",
        value="",
        placeholder="Paste only if not using Streamlit Secrets",
        help="Never commit your Groq API key to GitHub.",
    )

    api_key = manual_key.strip() or os.getenv("GROQ_API_KEY", "").strip() or secret_key

    st.divider()
    st.markdown(
        "**Source:** Sindh Shops and Commercial Establishment Act, 2015 "
        "(Sindh Act No. XII of 2016), from the supplied PDF."
    )
    st.info(
        "This is a document-grounded information tool, not a substitute for advice "
        "from a qualified lawyer or the competent labour authority."
    )


# ============================================================
# PDF DOWNLOAD + TEXT EXTRACTION
# ============================================================

@st.cache_data(show_spinner=False)
def download_pdf() -> bytes:
    """Download the source PDF from the supplied direct-download URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Sindh-Act-RAG/1.0)"
    }
    response = requests.get(PDF_URL, headers=headers, timeout=90)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    data = response.content

    # Basic validation so an HTML error page is not silently indexed as a PDF.
    if not data.startswith(b"%PDF"):
        raise ValueError(
            "The supplied URL did not return a PDF file. "
            "Please check the Google Drive download link."
        )

    return data


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = re.sub(r"-\s*\n\s*", "", text)  # de-hyphenate line-break words
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_section(text: str, current_section: str) -> str:
    """
    Find the latest numbered statutory section in the text.
    Examples: 7., 14.(1), 24(2), etc.
    """
    matches = re.findall(r"(?<!\d)(\d{1,2})\s*\.\s*(?=[A-Z(])", text)
    if matches:
        return matches[-1]
    return current_section


def make_chunks(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract page text and create overlapping chunks.
    Metadata includes page and best-effort section number.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: List[Dict[str, Any]] = []

    current_section = "Preamble / Definitions"

    for page_no, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        text = clean_text(raw)

        if not text:
            continue

        current_section = detect_section(text, current_section)

        # ~1000 characters keeps retrieval precise for this 17-page statute.
        chunk_size = 1100
        overlap = 180
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if len(chunk) >= 80:
                section = detect_section(chunk, current_section)
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_no,
                        "section": section,
                        "source": "Supplied Sindh Shops and Commercial Establishment Act PDF",
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    doc.close()
    return chunks


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

@st.cache_resource(show_spinner="Downloading source PDF, loading embeddings and building FAISS index...")
def build_retriever():
    pdf_bytes = download_pdf()
    chunks = make_chunks(pdf_bytes)

    if not chunks:
        raise ValueError("No readable text was extracted from the supplied PDF.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [item["text"] for item in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    top_k: int,
):
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_vector, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)

    return results


# ============================================================
# PROMPTING
# ============================================================

TECHNICALITY_INSTRUCTIONS = {
    "Simple": (
        "Explain in plain language. Define legal terms briefly. "
        "Assume the reader is not a lawyer."
    ),
    "Standard": (
        "Give a clear practical explanation with the relevant section numbers "
        "and important conditions."
    ),
    "Technical": (
        "Use precise statutory terminology, distinguish subsections and provisos, "
        "and explain the legal conditions carefully."
    ),
    "Legal/Expert": (
        "Use formal legal terminology. Analyze the retrieved statutory text closely, "
        "identify section/subsection relationships, provisos, exceptions and limitations, "
        "and avoid conclusions not supported by the source."
    ),
}

SIZE_INSTRUCTIONS = {
    "Short": "Answer in 2–5 concise bullet points.",
    "Medium": "Answer in a focused explanation, normally 1–4 short paragraphs or bullets.",
    "Long": "Give a detailed explanation with headings and relevant statutory provisions.",
    "Very long": (
        "Give a comprehensive explanation with headings, conditions, exceptions, "
        "practical implications, and source references. Do not add unsupported law."
    ),
}

LANGUAGE_INSTRUCTIONS = {
    "English": "Answer in English.",
    "Urdu": "Answer in Urdu script where practical; retain section numbers and important legal terms.",
    "Roman Urdu": "Answer in Roman Urdu; retain section numbers and important legal terms.",
}


def build_context(results: List[Dict[str, Any]]) -> str:
    blocks = []

    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {i} | Section {item['section']} | PDF page {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(blocks)


def answer_question(
    question: str,
    results: List[Dict[str, Any]],
    api_key: str,
    technicality: str,
    response_size: str,
    language: str,
    reasoning: str,
    temperature: float,
    strict_mode: bool,
):
    if not api_key:
        raise ValueError(
            "Groq API key is missing. Add GROQ_API_KEY to Streamlit Secrets, "
            "set it as an environment variable, or enter it in the sidebar."
        )

    client = Groq(api_key=api_key)
    context = build_context(results)

    source_rule = (
        "Use ONLY the supplied source passages. If the passages do not contain enough "
        "information, explicitly say: 'The supplied Act text does not provide enough "
        "information to answer this part.' Do not invent rules, fees, deadlines, "
        "exceptions, amendment effects, or procedures."
        if strict_mode
        else
        "Prefer the supplied source passages. If something is not in them, clearly label "
        "it as outside the supplied document rather than presenting it as a fact from the Act."
    )

    system_prompt = f"""
You are a source-grounded legal information assistant for the Sindh Shops and
Commercial Establishment Act document supplied to this application.

{source_rule}

{TECHNICALITY_INSTRUCTIONS[technicality]}
{SIZE_INSTRUCTIONS[response_size]}
{LANGUAGE_INSTRUCTIONS[language]}

Important rules:
1. Answer the user's exact question.
2. Cite the relevant statutory section and PDF page using the source labels, e.g.
   "Section 7, PDF page 6".
3. If multiple sections apply, distinguish them.
4. Preserve the document's wording and legal terminology where relevant.
5. Do not silently correct apparent typographical or drafting issues in the supplied PDF.
   If the text appears unclear, say so.
6. Do not claim that a provision is currently in force beyond what the supplied document
   itself establishes.
7. Do not use general Pakistani labour law, other Sindh laws, case law, websites, or
   assumptions unless the user explicitly asks for outside material.
8. Do not reveal hidden reasoning or internal chain-of-thought.
9. End with a brief "Source basis" line listing the key section(s) and PDF page(s).
"""

    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED SOURCE PASSAGES:
{context}

Now provide the answer.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        max_completion_tokens={
            "Short": 600,
            "Medium": 1200,
            "Long": 2200,
            "Very long": 3600,
        }[response_size],
        reasoning_effort=reasoning,
        include_reasoning=False,
    )

    return completion.choices[0].message.content


# ============================================================
# INITIALIZE RAG
# ============================================================

try:
    embedding_model, faiss_index, source_chunks = build_retriever()
except Exception as exc:
    st.error("The source document could not be downloaded or indexed.")
    st.exception(exc)
    st.stop()

st.success(
    f"RAG ready — {len(source_chunks)} searchable passages indexed from the supplied PDF."
)

# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

with st.expander("💡 Example questions"):
    st.markdown(
        """
- What is the maximum closing time for an establishment?
- How many hours can an adult employee work in a day and a week?
- What is the overtime wage rate?
- How many annual leave days are provided?
- What are the casual and sick leave entitlements?
- Can a child be employed in an establishment?
- What is the notice period for a permanent employee?
- When must an establishment be registered?
- What are the registration fees mentioned in section 24?
- What is the weekly holiday rule?
- What are the penalties for violating section 7?
- What powers does an Inspector have?
- Which establishments are excluded from the Act?
- What does the Act mean by "commercial establishment"?
"""
    )

# ============================================================
# CHAT
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about the Sindh Shops & Commercial Establishment Act..."
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving the relevant statutory passages and generating the answer..."):
            try:
                results = retrieve(
                    question,
                    embedding_model,
                    faiss_index,
                    source_chunks,
                    top_k,
                )

                answer = answer_question(
                    question=question,
                    results=results,
                    api_key=api_key,
                    technicality=technicality,
                    response_size=response_size,
                    language=language,
                    reasoning=reasoning,
                    temperature=temperature,
                    strict_mode=strict_mode,
                )

                st.markdown(answer)

                if show_sources:
                    with st.expander("📚 Retrieved source passages"):
                        for i, item in enumerate(results, start=1):
                            st.markdown(
                                f"**Source {i} — Section {item['section']} — "
                                f"PDF page {item['page']} — similarity {item['score']:.3f}**"
                            )
                            st.write(item["text"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error("The answer could not be generated.")
                st.exception(exc)

st.divider()
st.caption(
    "RAG pipeline: Google Drive PDF → PyMuPDF text extraction → overlapping chunks → "
    "Sentence Transformers embeddings → FAISS similarity search → Groq GPT-OSS 20B."
)
