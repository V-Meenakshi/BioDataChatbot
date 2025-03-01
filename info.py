import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fuzzywuzzy import process

# Configure Gemini securely (avoid hardcoding API keys in production)
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini = genai.GenerativeModel('gemini-1.5-flash')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv('my_data.csv')
    df.dropna(subset=['question', 'answer'], inplace=True)
    df['context'] = df.apply(
        lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}",
        axis=1
    )
    embeddings = embedder.encode(df['context'].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return df, index

df, faiss_index = load_data()

def find_closest_question(query, faiss_index, df):
    # First, try FAISS embedding search
    query_embedding = embedder.encode([query])
    D, I = faiss_index.search(query_embedding.astype('float32'), k=1)
    if len(I) > 0 and I[0][0] != -1:
        matched_idx = I[0][0]
        matched_question = df.iloc[matched_idx]['question']
        matched_answer = df.iloc[matched_idx]['answer']

        # Optional: If distance is too high, fallback to fuzzy
        if D[0][0] > 1.0:  # Adjust threshold as needed
            # Use fuzzy matching as fallback
            matched_answer = fuzzy_match_query(query, df)
        return matched_answer
    else:
        # If no match found, use fuzzy matching
        return fuzzy_match_query(query, df)

def fuzzy_match_query(query, df):
    questions = df["question"].tolist()
    best_match, score = process.extractOne(query, questions)
    if score >= 70:  # Adjust threshold
        return df[df["question"] == best_match]["answer"].values[0]
    return None

def generate_refined_answer(query, retrieved_answer):
    # Provide strong instructions to not override the CSV content
    prompt = f"""
    You are Meenakshi Vinjamuri, an AIML student.
    You must strictly use the following retrieved answer as your source of truth.
    
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    
    - DO NOT contradict the retrieved answer.
    - Provide a friendly and conversational response, but keep the factual data unchanged.
    - Keep it concise, clear, and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Streamlit Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Ask me anything about Meenakshi's bio...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        answer = find_closest_question(user_input, faiss_index, df)
        if answer:
            final_response = generate_refined_answer(user_input, answer)
        else:
            final_response = "I’m sorry, I don’t have an answer for that."

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.rerun()
