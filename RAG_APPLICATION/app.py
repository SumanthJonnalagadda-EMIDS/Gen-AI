import os
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

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = PyPDFLoader("framework.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
chunks = splitter.split_documents(docs)
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)
vectorstore = FAISS.from_documents(chunks, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """
You are a helpful assistant responding to a user with the role: {user_role}.
Use the information retrieved from the document to answer the following question.
Answer in a tone and format appropriate to that role.
Context:{context}
Question:{query}
Helpful Answer:
"""
prompt = PromptTemplate(
    input_variables=["query", "user_role", "context"],
    template=template
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.4, google_api_key=api_key
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

st.set_page_config(page_title="Healthcare RAG Assistant",
                   page_icon="💬", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>👩‍⚕️ Healthcare RAG Assistant</h1>",
    unsafe_allow_html=True
)

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

st.markdown("---")
question = st.text_input(
    "🔍 Ask your question",
    placeholder="e.g., What are key recommendations for managing TB and Diabetes?"
)
submit = st.button("Get Answer 💡", use_container_width=True)

if submit:
    if not question.strip():
        st.warning("Please enter a valid question.")
    elif "previous" in question.lower() and "question" in question.lower():
        history = st.session_state["conversations"][user_role]
        if history:
            st.markdown("### 🕘 Previous Questions & Answers")
            for i, chat in enumerate(history, start=1):
                st.markdown(f"**Q{i}:** {chat['user']}")
                st.markdown(f"**A{i}:** {chat['ai']}")
                st.markdown("---")
        else:
            st.info("No previous questions found.")
    else:
        try:
            docs = retriever.invoke(question)
            response = stuff_chain.invoke({
                "input_documents": docs,
                "query": question,
                "user_role": user_role
            })
            answer = response.get("output_text", "").strip()
            st.session_state["conversations"][user_role].append({
                "user": question,
                "ai": answer
            })
            st.markdown("###  Response")
            with st.container():
                for line in answer.split("\n"):
                    if line.strip():
                        st.write("🔹 " + line.strip())
        except ResourceExhausted:
            st.error(
                "Gemini API quota exceeded. Please upgrade to increase your limits.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if st.session_state["conversations"][user_role]:
    st.markdown("### 📜 Conversation History")
    for idx, msg in enumerate(st.session_state["conversations"][user_role]):
        st.markdown(f"** Me:** {msg['user']}")
        st.markdown(f"** AI:** {msg['ai']}")
        st.markdown("---")

if st.button("Clear 🧹", use_container_width=True):
    st.session_state["conversations"][user_role] = []
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
        else:
            st.warning(
                "Unable to refresh the app — rerun function not available.")
    except Exception as e:
        st.error(f"Could not rerun the app: {e}")
