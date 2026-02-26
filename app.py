import streamlit as st
import requests
import base64
import os

st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Dataset Chat Agent")
st.markdown("Ask questions about the Titanic dataset in natural language.")
backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


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
            f"{backend_url}/ask",
            json={"question": question},
            timeout=30
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
