import streamlit as st
import seaborn as sns

df = sns.load_dataset("titanic")

st.write(df.head())