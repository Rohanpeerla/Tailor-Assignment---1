import base64

import streamlit as st

from agent import run_query


st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon="T",
    layout="wide",
)

st.title("Titanic Dataset Chat Agent")
st.markdown("Ask questions about the Titanic dataset in natural language.")
st.caption("Streamlit-only mode: no separate backend deployment needed.")

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
        answer, image = run_query(question)
    except Exception as e:
        answer = f"Query processing error: {e}"
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
