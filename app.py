import base64
import os

import requests
import streamlit as st


def _resolve_backend_base_url() -> str:
    url = None
    try:
        url = st.secrets.get("BACKEND_URL")
    except Exception:
        url = None

    if not url:
        url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    url = str(url).strip().rstrip("/")
    if url.endswith("/ask"):
        url = url[:-4]
    return url


st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon="T",
    layout="wide",
)

st.title("Titanic Dataset Chat Agent")
st.markdown("Ask questions about the Titanic dataset in natural language.")

backend_url = _resolve_backend_base_url()
ask_url = f"{backend_url}/ask"
health_url = f"{backend_url}/health"

try:
    health = requests.get(health_url, timeout=6)
    if health.ok:
        st.caption(f"Backend connected: {backend_url}")
    else:
        st.warning(f"Backend reachable but unhealthy ({health.status_code}): {backend_url}")
except Exception:
    st.warning(f"Backend not reachable: {backend_url}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg and msg["image"]:
            image_bytes = base64.b64decode(msg["image"])
            st.image(image_bytes)

question = st.chat_input("Type your question here...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        response = requests.post(
            ask_url,
            json={"question": question},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "No response")
        image = data.get("image", None)
    except Exception as e:
        answer = f"Backend connection error ({backend_url}): {e}"
        image = None

    assistant_message = {"role": "assistant", "content": answer}
    if image:
        assistant_message["image"] = image
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if image:
            image_bytes = base64.b64decode(image)
            st.image(image_bytes)
