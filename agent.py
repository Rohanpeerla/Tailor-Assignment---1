import os
import io
import base64
import importlib
import re
import requests
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"


def _load_titanic_df() -> pd.DataFrame:
    try:
        return sns.load_dataset("titanic")
    except Exception:
        
        return pd.DataFrame(
            {
                "sex": ["male", "female", "female", "male"],
                "survived": [0, 1, 1, 1],
                "boat": [None, "4", "11", "7"],
                "fare": [7.25, 71.2833, 7.925, 8.05],
                "age": [22, 38, 26, 35],
                "embark_town": ["Southampton", "Cherbourg", "Southampton", "Southampton"],
            }
        )


df = _load_titanic_df()
agent = None
chat_llm = None

try:
    if GOOGLE_API_KEY:
        lc_agents = importlib.import_module("langchain_experimental.agents")
        lc_google = importlib.import_module("langchain_google_genai")

        chat_llm = lc_google.ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )

        agent = lc_agents.create_pandas_dataframe_agent(
            chat_llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
        )
except Exception:
    agent = None


def _gemini_dataset_answer(question: str):
    if not GOOGLE_API_KEY:
        return None

    preview_csv = df.head(40).to_csv(index=False)
    columns = ", ".join([f"{c} ({str(t)})" for c, t in df.dtypes.items()])

    prompt = f"""
You are a Titanic dataset analyst. Answer using only the provided dataframe context.
If the answer cannot be derived from the data context, say that clearly.
Give a concise answer and include computed values when relevant.

Question:
{question}

DataFrame facts:
- Rows: {len(df)}
- Columns: {columns}
- Sample rows (CSV):
{preview_csv}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join([p.get("text", "") for p in parts]).strip()
        return text if text else None
    except Exception:
        return None


def _extract_text_from_gemini_response(data: dict) -> str:
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join([p.get("text", "") for p in parts]).strip()


def _gemini_pandas_expression(question: str):
    if not GOOGLE_API_KEY:
        return None

    columns = ", ".join([f"{c} ({str(t)})" for c, t in df.dtypes.items()])
    prompt = f"""
You translate user questions about a pandas DataFrame named df into ONE valid Python pandas expression.
Rules:
- Output only the expression, no markdown, no explanation.
- Use only df, pd, np.
- The expression must evaluate to an answer (scalar/Series/DataFrame/string).
- Prefer exact computations from df.

DataFrame columns:
{columns}

Question:
{question}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        expr = _extract_text_from_gemini_response(response.json())
        expr = expr.strip().strip("`")
        if expr.lower().startswith("python"):
            expr = expr[6:].strip()
        return expr if expr else None
    except Exception:
        return None


def _evaluate_pandas_expression(expr: str):
    safe_globals = {"__builtins__": {}}
    safe_locals = {"df": df, "pd": pd, "np": np}
    result = eval(expr, safe_globals, safe_locals)

    if isinstance(result, pd.DataFrame):
        return result.to_string(index=False)
    if isinstance(result, pd.Series):
        return result.to_string()
    return str(result)


def _fallback_query(question: str):
    q = question.lower()
    image_base64 = None

    if re.search(r"\bfemale\b", q) and "percentage" in q:
        percent = (df["sex"].value_counts(normalize=True).get("female", 0.0)) * 100
        return f"{percent:.2f}% of passengers were female.", None

    if re.search(r"\bmale\b", q) and "percentage" in q:
        percent = (df["sex"].value_counts(normalize=True).get("male", 0.0)) * 100
        return f"{percent:.2f}% of passengers were male.", None

    if "average" in q and "fare" in q:
        return f"The average ticket fare was {df['fare'].mean():.2f}.", None

    if "surviv" in q and "how many" in q:
        if "boat" in q and "boat" in df.columns:
            survived_in_boat = int(((df["survived"] == 1) & df["boat"].notna()).sum())
            return f"{survived_in_boat} passengers survived and have a boat number recorded.", None

        survived_count = int((df["survived"] == 1).sum())
        return f"{survived_count} passengers survived.", None

    if "embark" in q:
        return df["embark_town"].value_counts().to_string(), None

    if "histogram" in q and "age" in q:
        plt.figure()
        df["age"].dropna().hist(bins=20)
        plt.title("Age Distribution")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return "Here is the histogram of passenger ages.", image_base64

    return None


def run_query(question: str):
    # Keep chart output deterministic so we can return an image payload.
    q = question.lower()
    if "histogram" in q and "age" in q:
        chart_answer = _fallback_query(question)
        if chart_answer is not None:
            return chart_answer

    if agent is not None:
        try:
            response = agent.run(question)
            if response:
                return str(response), None
        except Exception:
            pass

    expr = _gemini_pandas_expression(question)
    if expr:
        try:
            computed = _evaluate_pandas_expression(expr)
            if computed:
                return computed, None
        except Exception:
            pass

    gemini_response = _gemini_dataset_answer(question)
    if gemini_response:
        return gemini_response, None

    fallback_answer = _fallback_query(question)
    if fallback_answer is not None:
        return fallback_answer

    return "I couldn't compute that from the local rules. Install LangChain deps or keep a valid GOOGLE_API_KEY for full dataset Q&A.", None
