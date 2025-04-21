
import streamlit as st
st.set_page_config(page_title="MedBot", page_icon="💊")
import re
from datetime import datetime
from dotenv import load_dotenv
import os 
import sqlite3
import pandas as pd
import numpy as np
import google.generativeai as genai

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


# ============================ Load Env & API ============================
load_dotenv()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    raise ValueError("No GOOGLE_API_KEY in .env")
genai.configure(api_key=GOOGLE_API_KEY)

# ============================ File Paths ============================
FORECAST_DB = "forecasts.db"
INV_CSV = "fake_medicine_inventory.csv"
USAGE_CSV = "sample_drug_usage_with_dates.csv"

usage_df = pd.read_csv(USAGE_CSV, parse_dates=["date"])
inv_df = pd.read_csv(INV_CSV, parse_dates=True)

DRUG_COL = next((c for c in inv_df.columns if "drug" in c.lower()), None)
MFG_COL = next((c for c in inv_df.columns if "manufactur" in c.lower() or "mfg" in c.lower()), None)
EXP_COL = next((c for c in inv_df.columns if "expiry" in c.lower() or "exp" in c.lower()), None)

_drug_list = sorted(
    set(inv_df[DRUG_COL].str.lower()) |
    set(c.lower() for c in usage_df.columns if c.lower() != "date"),
    key=lambda s: -len(s)
)

def extract_drug(text: str) -> str:
    txt = text.lower()
    for d in _drug_list:
        if re.search(rf"\b{re.escape(d)}\b", txt):
            return d
    return None

# ============================ LLM Wrapper ============================
class GoogleGenerativeLLM(LLM):
    model_name: str = "gemini-2.0-flash"
    max_output_tokens: int = 750

    @property
    def _llm_type(self): return "google_generative"
    @property
    def _identifying_params(self): return {"model_name": self.model_name, "max_output_tokens": self.max_output_tokens}

    def _call(self, prompt: str, stop=None) -> str:
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content(prompt)
        if not getattr(resp, "text", None):
            raise ValueError("No response from Google Generative AI.")
        return resp.text

def direct_llm_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    r = model.generate_content(prompt)
    return getattr(r, "text", str(r))

# ============================ Document Loaders ============================
def load_inventory_documents_sql():
    conn = sqlite3.connect(FORECAST_DB)
    df = pd.read_sql_query("SELECT * FROM forecasts", conn)
    conn.close()
    return [Document(page_content=(
        f"Drug: {r['drug']}\n"
        f"Current Inventory: {r['current_inventory']}\n"
        f"Forecast: {r['forecast_day1']}, {r['forecast_day2']}, {r['forecast_day3']}\n"
        f"Total Forecast: {r['total_forecast']}\n"
        f"Reorder Qty: {r['reorder_quantity']}\n"
    )) for _, r in df.iterrows()]

def load_inventory_documents_csv():
    return [Document(page_content=(
        f"Drug: {r[DRUG_COL]}\n"
        f"Manufacturing Date: {pd.to_datetime(r[MFG_COL]).strftime('%Y-%m-%d')}\n"
        f"Expiry Date: {pd.to_datetime(r[EXP_COL]).strftime('%Y-%m-%d')}\n"
    )) for _, r in inv_df.iterrows()]

def load_usage_documents_with_dates():
    docs = []
    for c in usage_df.columns:
        if c.lower() == "date": continue
        text = "".join(f"{r['date'].strftime('%Y-%m-%d')}: {r[c]}\n" for _, r in usage_df.iterrows())
        docs.append(Document(page_content=f"Usage for {c}:\n{text}"))
    return docs

def build_vector_store():
    docs = []
    docs += load_inventory_documents_sql()
    docs += load_inventory_documents_csv()
    docs += load_usage_documents_with_dates()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, emb)

# ============================ Handlers ============================
def check_reorder_requirement(drug):
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT reorder_quantity FROM forecasts WHERE LOWER(drug)=?", (drug,))
    row = cur.fetchone(); conn.close()
    if not row: return f"No reorder data for '{drug}'."
    qty = row[0]
    return f"Reorder needed: {'Yes' if qty > 0 else 'No'}. Quantity: {int(qty)} units."

def check_forecast(drug):
    conn = sqlite3.connect(FORECAST_DB)
    cur = conn.cursor()
    cur.execute("SELECT forecast_day1,forecast_day2,forecast_day3,total_forecast FROM forecasts WHERE LOWER(drug)=?", (drug,))
    row = cur.fetchone(); conn.close()
    if not row: return f"No forecast data for '{drug}'."
    d1, d2, d3, total = row
    return f"Forecast for {drug}:\n • Day 1: {d1}\n • Day 2: {d2}\n • Day 3: {d3}\n • Total: {total}"

def check_usage_by_date(drug, date_str):
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y").date()
    except ValueError:
        return "Date must be in dd/mm/yyyy format."
    mask = usage_df["date"].dt.date == dt
    row = usage_df.loc[mask]
    if row.empty:
        return f"No usage data for '{drug}' on {date_str}."
    usage_col = next((c for c in usage_df.columns if c.strip().lower() == drug.strip().lower()), None)
    if not usage_col:
        return f"No usage data for '{drug}'."
    val = row.iloc[0][usage_col]
    return f"Usage for {drug} on {date_str}: {val}"

def check_usage_by_day(drug, day):
    if day < 1 or day > len(usage_df):
        return f"Day {day} out of range (1–{len(usage_df)})."
    row = usage_df.iloc[day - 1]
    date_str = row["date"].strftime("%d/%m/%Y")
    usage_col = next((c for c in usage_df.columns if c.lower() == drug), None)
    if not usage_col:
        return f"No usage data for '{drug}'."
    val = row[usage_col]
    return f"Day {day} ({date_str}) usage for {drug}: {val}"

def check_mfg_date(drug):
    sub = inv_df[inv_df[DRUG_COL].str.lower() == drug]
    if sub.empty:
        return f"No manufacturing data for '{drug}'."
    return f"Manufacturing date for {drug}: {pd.to_datetime(sub.iloc[0][MFG_COL]).strftime('%d/%m/%Y')}"

def check_expiry_date(drug):
    sub = inv_df[inv_df[DRUG_COL].str.lower() == drug]
    if sub.empty:
        return f"No expiry data for '{drug}'."
    return f"Expiry date for {drug}: {pd.to_datetime(sub.iloc[0][EXP_COL]).strftime('%d/%m/%Y')}"

def is_general_medical_query(q):
    ql = q.lower()
    kws = [
        "symptom", "treat", "treatment", "diet", "cure", "prevent", "body", "pain", "headache", "bodyache"
        "prevention", "medicine", "medication", "drug", "salt formula",
        "formula", "use", "dosage", "salt", "composition", "chemical", "consult", "cure", "health", "disease", "accident", "injury", "weight", "gain", "loss", "bones"
    ]
    return any(kw in ql for kw in kws)

# ============================ LangChain Setup ============================
@st.cache_resource
def setup_chain():
    memory = ConversationBufferMemory(memory_key="chat_history")
    vs = build_vector_store()
    llm = GoogleGenerativeLLM()
    prompt = PromptTemplate.from_template(
        """
You are a medical domain expert. Answer only queries about:
- drug usage, inventory, reorder needs, side effects, formulas.
Reject any non-medical questions.

Context:
{context}

Question: {question}

Answer (structured, no * or $):
"""
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k":4}),
        return_source_documents=False,
        memory=memory,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

qa_chain = setup_chain()

# ============================ Streamlit Chat Interface ============================
#st.set_page_config(page_title="MedBot", page_icon="💊")
st.title("💊 MedBot - AI Medical Assistant")
col1, col2 = st.columns([0.8, 0.2])
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything...")

if user_input:
    ql = user_input.lower()
    drug = extract_drug(user_input)
    date_m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", user_input)
    day_m = re.search(r"(\d+)(?:st|nd|rd|th)?\s*day\b", ql)

    if "reorder" in ql and drug:
        answer = check_reorder_requirement(drug)
    elif date_m and drug:
        answer = check_usage_by_date(drug, date_m.group(1))
    elif day_m and drug:
        answer = check_usage_by_day(drug, int(day_m.group(1)))
    elif ("manufactur" in ql or "mfg" in ql) and drug:
        answer = check_mfg_date(drug)
    elif "expir" in ql and drug:
        answer = check_expiry_date(drug)
    elif "forecast" in ql and drug:
        answer = check_forecast(drug)
    elif is_general_medical_query(user_input):
        raw = direct_llm_response(
            "You are a specialist medical assistant. Provide a clear, structured answer "
            "without special characters (*, $, etc.).\n"
            f"Question: {user_input}\nAnswer:\n"
        )
        answer = re.sub(r"[\*\$]", "", raw).strip()
    else:
        answer = "Sorry, I can only answer medical-related queries."

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

# Display chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

