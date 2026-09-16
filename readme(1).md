# Sindh Shops & Commercial Establishment Act — RAG Assistant

A source-grounded Retrieval-Augmented Generation (RAG) application built with:

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq `openai/gpt-oss-20b`

The application downloads the supplied Sindh Shops and Commercial Establishment Act PDF from the configured Google Drive direct-download URL when the RAG resources are first initialized, extracts the PDF text, creates embeddings, builds a FAISS vector index, retrieves the most relevant passages, and sends only those passages to Groq for the final answer.

## Important source scope

The application is intentionally designed around the supplied PDF:

**The Sindh Shops and Commercial Establishment Act, 2015 (Sindh Act No. XII of 2016)** and the amendment material contained in the supplied document.

The app does not silently replace the supplied document with another online copy. If the retrieved source does not contain enough information, the default **Strict source mode** instructs the assistant to say so rather than inventing an answer.

This is a legal-information/RAG demonstration, not legal advice.

## Files

Only these three files are required:

```text
app.py
requirements.txt
readme.md
```

No local PDF needs to be committed to GitHub because `app.py` downloads the supplied PDF URL at startup.

## Features

### RAG pipeline

```text
Supplied PDF
    ↓
Google Drive download
    ↓
PyMuPDF
    ↓
Text cleaning
    ↓
Overlapping chunks
    ↓
Sentence Transformers
    ↓
Normalized embeddings
    ↓
FAISS IndexFlatIP
    ↓
Top-K retrieval
    ↓
Groq GPT-OSS 20B
    ↓
Source-grounded answer
```

### UI controls

The sidebar provides:

- **Technicality**
  - Simple
  - Standard
  - Technical
  - Legal/Expert

- **Response size**
  - Short
  - Medium
  - Long
  - Very long

- **Answer language**
  - English
  - Urdu
  - Roman Urdu

- **Retrieved passages**
  - Controls FAISS Top-K

- **Creativity / strictness**
  - Temperature control
  - A low value is recommended for statutory questions

- **Reasoning effort**
  - low
  - medium
  - high

- **Show retrieved source passages**
  - Displays the actual passages used for the answer

- **Strict source mode**
  - Forces the model to stay inside the supplied document

## Groq API key

Do **not** put your Groq API key inside `app.py`.

### Streamlit Cloud

After creating the Streamlit app:

1. Open your Streamlit app.
2. Open **Settings**.
3. Open **Secrets**.
4. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Save the secret and restart/reboot the app if necessary.

### Colab / local testing

You can set an environment variable:

```python
import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

The application also contains an optional password field in the sidebar so you can paste a key during a temporary test without editing the source code.

Never commit a real API key to GitHub.

## GitHub → Streamlit Cloud deployment

### Step 1 — Create a GitHub repository

Create a new GitHub repository, for example:

```text
sindh-shops-act-rag
```

You only need to upload:

```text
app.py
requirements.txt
readme.md
```

Do not upload:

- `.env`
- your Groq API key
- large generated FAISS files
- the PDF
- Python virtual environments

### Step 2 — Upload the three files

On GitHub:

1. Open the repository.
2. Click **Add file** → **Create new file**.
3. Create `app.py` and paste the application code.
4. Create `requirements.txt` and paste the dependency list.
5. Create `readme.md` and paste this README.
6. Commit the changes.

### Step 3 — Create the Streamlit app

Open Streamlit Community Cloud and sign in with GitHub.

Create a new app and select:

```text
Repository: your GitHub repository
Branch: main
Main file: app.py
```

Deploy.

### Step 4 — Add the Groq secret

In Streamlit Cloud:

```text
App
→ Settings
→ Secrets
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Save.

The app will then download the PDF and build the RAG index.

## Colab testing

The application can also be run from Google Colab.

Install the requirements:

```python
!pip install -r requirements.txt
```

Set the Groq key:

```python
import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

Then run Streamlit using your preferred Colab Streamlit/tunneling setup.

The RAG application itself does not require a paid vector database or paid embedding service.

## Why FAISS + Sentence Transformers?

This project uses local embeddings instead of an external embedding API.

`all-MiniLM-L6-v2` creates the document and query embeddings locally.

FAISS then performs vector similarity search locally.

This keeps the RAG retrieval layer inexpensive and simple enough for a beginner-friendly Streamlit deployment.

## Why the application uses source passages

A legal RAG system should not simply ask an LLM:

> "What does the Sindh Shops Act say?"

Instead, the application:

1. Converts the statute into searchable chunks.
2. Embeds those chunks.
3. Finds passages semantically related to the question.
4. Supplies those passages to the LLM.
5. Instructs the LLM to answer from those passages.
6. Shows the retrieved passages so the user can inspect the evidence.

This reduces unsupported answers, although no RAG system can guarantee perfect legal interpretation.

## Example questions

Try:

```text
What is the maximum closing time for an establishment?
```

```text
How many hours can an adult employee work per day?
```

```text
What is the overtime rate?
```

```text
How many days of annual leave are provided?
```

```text
What are the casual and sick leave entitlements?
```

```text
Can a child be employed in an establishment?
```

```text
What notice is required to terminate a permanent employee?
```

```text
When must an establishment be registered?
```

```text
What registration fees are stated in section 24?
```

```text
What is the weekly holiday rule?
```

```text
What are the penalties for violating section 7?
```

```text
What powers does an Inspector have?
```

## Accuracy design

The application uses several controls specifically for statutory Q&A:

### Low default temperature

The default temperature is `0.1` because legal/statutory questions generally benefit from consistent, source-focused responses rather than creative generation.

### Source citations

Retrieved passages contain:

```text
Section number
PDF page
similarity score
```

The model is instructed to reference relevant sections and PDF pages.

### Strict source mode

Strict mode tells the model not to manufacture:

- fees
- deadlines
- exceptions
- penalties
- eligibility rules
- amendment effects
- procedures

when those details are not supported by the retrieved source.

### User-visible retrieval

The application can show the exact retrieved passages, making it easier to inspect what the model received.

## Notes about startup time

The first startup can take longer because the application may need to:

1. Download the PDF.
2. Load the Sentence Transformer model.
3. Extract the PDF.
4. Generate embeddings.
5. Build the FAISS index.

Streamlit caching is used so the expensive PDF/embedding/index work is reused during the app's active process instead of being repeated on every question.

If the Streamlit app is restarted or its cache is cleared, the initialization process runs again.

## Free-tier considerations

The application itself uses open-source Python libraries for PDF processing, embeddings, and FAISS retrieval.

Groq API usage is subject to the limits and availability of the Groq account/model you use. Streamlit Community Cloud also has its own resource limits.

The code does not require:

- OpenAI API
- Pinecone
- Weaviate
- Chroma Cloud
- a paid database
- a paid embedding API

## Troubleshooting

### `ModuleNotFoundError: No module named 'faiss'`

Make sure `requirements.txt` contains:

```text
faiss-cpu>=1.10.0
```

Then redeploy/reboot the Streamlit app.

### PDF download error

The application validates that the URL actually returns a PDF.

If the Google Drive URL stops working, replace `PDF_URL` in `app.py` with a valid direct-download URL for the same source document.

### Groq API key missing

Check:

```text
Streamlit Cloud
→ Settings
→ Secrets
```

and ensure:

```toml
GROQ_API_KEY = "..."
```

is present.

### Model error

The application currently uses:

```text
openai/gpt-oss-20b
```

If Groq changes model availability in the future, update `MODEL_NAME` in `app.py` to an available Groq chat model.

## Legal/source limitation

This application is a retrieval and explanation layer over the supplied PDF. It should not be treated as an official legal interpretation.

For an important employment or compliance decision, verify the applicable law, notifications, rules, amendments, official gazette material, and professional legal advice as appropriate.

## License

For educational/project use. Add an appropriate open-source license if you intend to publish the repository for reuse.
