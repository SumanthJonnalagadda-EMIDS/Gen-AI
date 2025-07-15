import os
import math
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from google.api_core.exceptions import ResourceExhausted
import matplotlib.pyplot as plt

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = PyPDFLoader("RAG_APPLICATION/framework.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
main_prompt = PromptTemplate(
    input_variables=["query", "user_role", "context"],
    template="""
You are a helpful, knowledgeable, and honest AI assistant. The user has the role: {user_role}.
You are answering a healthcare-related question using ONLY the provided document context. Avoid making assumptions or using external knowledge.
If a brief explanation or simplification is asked, do so using the same document context.
If the answer is not clearly present in the context, say: "The provided documents do not contain this information clearly."

User Role: {user_role}  
Context: {context}  
Question: {query}

Answer (validated from context):
"""
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.4, google_api_key=api_key)
llm_chain = LLMChain(llm=llm, prompt=main_prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="context")
geval_prompt = PromptTemplate(
    input_variables=["query", "chunk"],
    template="""
You are an evaluator. Based on the question and the chunk, rate the chunk's relevance to the question from 1 to 10.
Question:
{query}
Chunk:
{chunk}
Respond with only the score, as a number between 1 (not relevant) and 10 (very relevant).
"""
)
geval_chain = LLMChain(llm=llm, prompt=geval_prompt)
response_eval_prompt = PromptTemplate(
    input_variables=["query", "chunk"],
    template="""
You are a critical evaluator reviewing an AI-generated answer to a user's healthcare-related question.
Evaluate based on this context, scoring 1–10 for each metric. Respond with ONLY numbers, no explanations, and follow the format exactly.

Format:
Relevance: <1-10>  
Coherence: <1-10>  
Fluency: <1-10>  
Factual Accuracy: <1-10>  
Completeness: <1-10>  
Lack of Hallucination: <1-10>  
Validation Against Context: <1-10>  
Confidence Score: <average of above, rounded to 2 decimals>
"""
)
response_eval_chain = LLMChain(llm=llm, prompt=response_eval_prompt)


def calculate_confidence_softmax(eval_scores: dict) -> float:
    keys = [
        "Relevance", "Coherence", "Fluency",
        "Factual Accuracy", "Completeness",
        "Lack of Hallucination", "Validation Against Context"
    ]
    values = []
    for key in keys:
        try:
            values.append(int(eval_scores.get(key, 0)))
        except:
            values.append(0)
    exp_scores = [math.exp(v) for v in values]
    total = sum(exp_scores)
    probabilities = [e / total for e in exp_scores]
    confidence = sum([s * p for s, p in zip(values, probabilities)])
    return round(confidence, 2)


st.set_page_config(page_title="Healthcare RAG Assistant",
                   page_icon="💬", layout="wide")
st.markdown("<h1 style='text-align: center;'>👩‍⚕️ Healthcare RAG Assistant</h1>",
            unsafe_allow_html=True)
st.sidebar.header("Select Role")
roles = [
    "Doctor", "Nurse", "TB Program Manager", "Diabetes Program Manager",
    "Public Health Official", "Health Policy Maker", "Researcher", "Patient"
]
user_role = st.sidebar.radio("Role", roles)
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {}
if user_role not in st.session_state["conversations"]:
    st.session_state["conversations"][user_role] = []
question = st.text_input(
    "🔍 Ask your question", placeholder="e.g., What are key recommendations for managing TB and Diabetes?")
submit = st.button("Get Answer 💡", use_container_width=True)
if submit:
    if not question.strip():
        st.warning("Please enter a valid question.")
    elif "previous" in question.lower() and "question" in question.lower():
        history = st.session_state["conversations"][user_role]
        if history:
            st.markdown("### 🕘 Previous Questions & Answers")
            for i, chat in enumerate(history, 1):
                st.markdown(f"**Q{i}:** {chat['user']}")
                st.markdown(f"**A{i}:** {chat['ai']}")
                st.markdown("---")
        else:
            st.info("No previous questions found.")
    else:
        with st.spinner("🔄 Generating response..."):
            try:
                docs = retriever.invoke(question)
                scored_chunks = []
                for doc in docs:
                    chunk = doc.page_content
                    score_response = geval_chain.invoke(
                        {"query": question, "chunk": chunk})
                    try:
                        score = int(score_response["text"].strip())
                    except:
                        score = 0
                    scored_chunks.append((score, chunk))
                scored_chunks.sort(reverse=True, key=lambda x: x[0])
                top_chunk = scored_chunks[0][1]

                response = llm_chain.invoke({
                    "query": question,
                    "user_role": user_role,
                    "context": top_chunk
                })
                answer = response.get("text", "").strip()
                eval_scores = {}
                if not answer.strip():
                    st.warning("Empty answer, skipping evaluation.")
                else:
                    try:
                        score_response = response_eval_chain.invoke({
                            "query": question,
                            "chunk": answer
                        })
                        st.write("LLM evaluation output:",
                                 score_response["text"])  # Debug
                        lines = score_response["text"].strip().split("\n")
                        for line in lines:
                            if ":" in line:
                                key, value = line.split(":", 1)
                                value = value.strip().replace("/10", "")

                                if value.replace('.', '', 1).isdigit():
                                    eval_scores[key.strip()] = value

                        if all(k in eval_scores for k in [
                            "Relevance", "Coherence", "Fluency",
                            "Factual Accuracy", "Completeness",
                            "Lack of Hallucination", "Validation Against Context"
                        ]):
                            eval_scores["Confidence Score (Softmax)"] = str(
                                calculate_confidence_softmax(eval_scores))
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")
                st.session_state["conversations"][user_role].append({
                    "user": question,
                    "ai": answer
                })
                st.markdown("### 💡 Response")
                for line in answer.split("\n"):
                    if line.strip():
                        st.write("🔹 " + line.strip())
                metrics = [
                    "Relevance", "Coherence", "Fluency", "Factual Accuracy",
                    "Completeness", "Lack of Hallucination", "Validation Against Context"
                ]
                if eval_scores and all(m in eval_scores for m in metrics):
                    st.markdown("### 📊 Answer Evaluation & Confidence")
                    cols = st.columns(len(metrics) + 1)
                    for i, m in enumerate(metrics):
                        cols[i].metric(label=m, value=f"{eval_scores[m]}/10")
                    cols[-1].metric(label="Confidence Score",
                                    value=f"{eval_scores.get('Confidence Score', 'N/A')}/10")
                    st.markdown("#### Evaluation Metric Scores")
                    for m in metrics:
                        st.markdown(f"- **{m}:** {eval_scores[m]}/10")
                    st.markdown(
                        f"- **Confidence Score:** {eval_scores.get('Confidence Score', 'N/A')}/10")
                    if "Confidence Score (Softmax)" in eval_scores:
                        st.markdown(
                            f"- **Softmax Confidence Score:** {eval_scores['Confidence Score (Softmax)']}/10")
                    try:
                        values = [float(eval_scores[m]) for m in metrics]
                        confidence_score = float(
                            eval_scores["Confidence Score"])
                        fig, ax = plt.subplots(figsize=(8, 5))
                        bars = ax.bar(metrics, values, color="skyblue")
                        ax.set_ylim([0, 10])
                        ax.set_ylabel("Score (out of 10)")
                        ax.set_title("AI Answer Evaluation Metrics")
                        plt.xticks(rotation=30, ha="right")
                        ax.axhline(confidence_score, color="orange", linestyle="--",
                                   label=f"Confidence Score: {confidence_score}")
                        ax.legend()
                        for bar, value in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2, value + 0.2,
                                    f"{value:.2f}", ha='center', va='bottom', fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.markdown(
                            f"**✅ Confidence Score:** {confidence_score}/10"
                        )
                    except Exception as e:
                        st.warning(f"Could not plot evaluation metrics: {e}")
                else:
                    st.warning("No valid evaluation metrics to display.")

                st.markdown("### 📄 Retrieved Context Chunks (Top 5)")
                for i, (score, chunk) in enumerate(scored_chunks, 1):
                    st.markdown(f"**Chunk {i} - Score: {score}/10**")
                    st.info(chunk[:600])

            except ResourceExhausted:
                st.error("🚫 Gemini API quota exceeded.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
if st.session_state["conversations"][user_role]:
    st.markdown("### 📜 Conversation History")
    for idx, msg in enumerate(st.session_state["conversations"][user_role]):
        st.markdown(f"**Me:** {msg['user']}")
        st.markdown(f"**AI:** {msg['ai']}")
        st.markdown("---")
if st.button("Clear 🪩", use_container_width=True):
    st.session_state["conversations"][user_role] = []
    try:
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Could not rerun the app: {e}")
