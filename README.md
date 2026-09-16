Sindh Shops & Commercial Establishment Act — RAG Assistant

A Streamlit RAG application for question answering over the supplied Sindh Shops and Commercial Establishment Act, 2015 (Sindh Act No. XII of 2016) PDF.

Tech stack

Python

Streamlit

FAISS

Sentence Transformers (all-MiniLM-L6-v2)

PyMuPDF

Groq (openai/gpt-oss-20b)

Google Drive + gdown for automatic source-PDF download

Source PDF and download behavior

The app uses the official Sindh Laws PDF as its primary source URL:

https://www.sindhlaws.gov.pk/setup/publications_SindhCode/PUB-NEW-18-000109.pdf

This is the published Sindh Act No. XII of 2016 — The Sindh Shops and Commercial Establishment Act, 2015. The user's Google Drive file ID is also configured as a fallback.

The Google Drive share link you supplied is:

https://drive.google.com/file/d/1O_CjQSmShfJovVV9sU6NQ_mqZPFll17Q/view?usp=drive_link

If you want the app to use the Drive copy instead of the official Sindh Laws copy, make the Drive file public:

Open the PDF in Google Drive.

Click Share.

Under General access, select Anyone with the link.

Set the role to Viewer.

Save.

This matters because Streamlit Cloud does not have your personal Google login session. gdown can handle Google Drive share links and confirmation pages, but the file must be accessible to the unauthenticated application.

Why the old version failed

The /file/d/.../view Google Drive URL is a preview/share page, not guaranteed raw PDF bytes. Google can return HTML instead of the PDF. The old application checked the first bytes for %PDF and therefore raised the error.

The fixed application now:

Tries the official Sindh Laws PDF first.

Tries the supplied Google Drive share URL.

Tries Google Drive direct-download endpoints.

Falls back to gdown, which is designed to handle Google Drive confirmation/interstitial pages.

Validates the downloaded content with the PDF %PDF signature before PyMuPDF/FAISS indexing.

This means an HTML Drive permission page will never be passed to the embedding pipeline.

Files

Keep exactly these three files in the GitHub repository:

app.py
requirements.txt
readme.md

Do not commit the PDF or your Groq API key.

Groq API key

Streamlit Cloud

After deploying:

App → Settings → Secrets

Add:

GROQ_API_KEY = "your_groq_api_key_here"

Save and reboot the app.

Local/Colab

You can either set GROQ_API_KEY as an environment variable or paste the key into the optional field in the sidebar.

Deploy through GitHub → Streamlit Community Cloud

Create a GitHub repository.

Upload only app.py, requirements.txt, and readme.md.

Open Streamlit Community Cloud.

Create a new app.

Select your GitHub repository.

Select the branch containing the files.

Set the main file to app.py.

Deploy.

Open Settings → Secrets and add GROQ_API_KEY.

Reboot/redeploy the application.

On the first startup, the app downloads the Google Drive PDF, extracts its text, creates Sentence Transformer embeddings, and builds the FAISS index. Streamlit caching prevents rebuilding the index on every normal rerun.

Features

The sidebar provides controls for:

Technicality: Simple / Standard / Technical / Legal/Expert

Response size: Short / Medium / Long / Very long

Answer language: English / Urdu / Roman Urdu

Number of retrieved passages

Temperature

Groq reasoning effort

Strict source mode

Display retrieved source passages

Source-grounded behavior

The assistant is instructed to answer from the supplied document and to identify when the retrieved source does not contain enough information. It also displays the retrieved passages so the user can inspect the basis of an answer.

The supplied document identifies itself as the Sindh Shops and Commercial Establishment Act, 2015, Sindh Act No. XII of 2016, and describes its purpose as amending and consolidating law concerning hours and other conditions of work and employment in establishments in Sindh.

Common questions to test

What is the maximum closing time for an establishment?

How many hours can an adult employee work per day and per week?

What is the overtime rate?

What is the weekly holiday rule?

How much annual leave is provided?

How much casual and sick leave is provided?

Can a child be employed?

What is the notice period for a permanent employee?

When must an establishment be registered?

What registration fees are stated in section 24?

What powers does an Inspector have?

What penalties are provided for violations?

Which establishments are excluded from the Act?

What is a commercial establishment?

Troubleshooting

Error: The source document could not be downloaded or indexed

Check the following:

The Google Drive file is set to Anyone with the link → Viewer.

The Drive file still has the same file ID:
1O_CjQSmShfJovVV9sU6NQ_mqZPFll17Q

gdown appears in requirements.txt.

Reboot/redeploy the Streamlit app after changing the repository.

Error: ModuleNotFoundError: faiss

Make sure faiss-cpu is present in requirements.txt, then reboot the Streamlit app so dependencies are reinstalled.

Error: Groq API key missing

Add GROQ_API_KEY under Streamlit Cloud Settings → Secrets, or enter the key in the sidebar.

Error: Groq rate limit

Groq API limits can be reached on free/low-volume accounts. Wait and retry, reduce response size, or lower the requested reasoning/output level.
