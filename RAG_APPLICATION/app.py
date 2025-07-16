import logging
from typing import Tuple, Dict, List
import os
import streamlit as st
import re
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# This import was in your original code
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from google.api_core.exceptions import ResourceExhausted

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

try:
    loader = PyPDFLoader("RAG_APPLICATION/framework.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
except FileNotFoundError:
    st.error("Error: 'Collaborative Framework for Care and Control.pdf' not found. Please ensure the PDF is in the same directory as the script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading or processing PDF: {e}")

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareGuardrails:
    """Custom guardrails implementation that works with your existing code"""

    def __init__(self):
        # Terms that should trigger content filtering
        self.restricted_terms = [
            "suicide", "self-harm", "overdose", "illegal drug",
            "harm yourself", "dangerous", "lethal", "deadly",
            "not approved", "experimental", "unproven"
        ]

        # Medical jargon that needs simplification for patients
        self.medical_jargon = [
            "comorbidity", "intervention", "glycemic control",
            "pathophysiology", "epidemiology", "pharmacological"
        ]

        # Roles that require simplified language
        self.simplify_for_roles = ["Patient", "Family Member"]

    def validate_response(self, text: str, user_role: str) -> Tuple[bool, str]:
        """
        Validates the response against healthcare safety rules
        Returns (is_valid, validation_message)
        """
        # Check for harmful content
        if self._contains_restricted_content(text):
            return False, "Response contains restricted medical content"

        # Check for medical jargon for non-professional roles
        if user_role in self.simplify_for_roles and self._contains_complex_terms(text):
            return False, "Response contains complex terms needing simplification"

        # Check response length is reasonable
        if not (50 <= len(text) <= 2000):
            return False, "Response length is not appropriate"

        return True, "Response passed all guardrails"

    def _contains_restricted_content(self, text: str) -> bool:
        """Check for restricted terms (case-insensitive)"""
        return any(term.lower() in text.lower() for term in self.restricted_terms)

    def _contains_complex_terms(self, text: str) -> bool:
        """Check for complex medical terms (case-insensitive)"""
        return any(term.lower() in text.lower() for term in self.medical_jargon)


# Initialize guardrails
guardrails = HealthcareGuardrails()


def filter_lifestyle_chunks(all_chunks, keywords=None):
    """
    Filters document chunks based on the presence of specified keywords related to lifestyle.
    """
    if keywords is None:
        keywords = ["lifestyle", "diet", "nutrition",
                    "exercise", "behavior", "physical activity"]
    return [doc for doc in all_chunks if any(kw in doc.page_content.lower() for kw in keywords)]


def simplify_for_role(text, user_role):
    """
    Simplifies medical terminology in the text based on the user's role,
    especially for 'Patient'.
    """
    if user_role == "Patient":
        terms = {
            "comorbidity": "other health problems",
            "intervention": "treatment",
            "glycemic control": "control of blood sugar",
            "adherence": "following medical advice",
            "diagnostic": "medical test",
            "etiology": "cause",
            "prognosis": "outlook or expected course",
            "pharmacological": "medication-related",
            "pathophysiology": "how the disease affects the body",
            "epidemiology": "how diseases spread in populations",
            "morbidity": "sickness or disease",
            "mortality": "death",
            "prevalence": "how common it is",
            "incidence": "rate of new cases",
            "screening": "checking for a disease",
            "compliance": "following advice",
            "therapeutic": "treatment-related"
        }
        for term, repl in terms.items():
            # Use regex for whole word replacement to avoid partial matches
            # \b ensures whole word match, e.g., 'adherence' but not 'nonadherence'
            text = re.sub(r'\b' + re.escape(term) + r'\b',
                          repl, text, flags=re.IGNORECASE)
    return text


# --- Original extract_eval_scores_markdown function restored ---
def extract_eval_scores_markdown(markdown_text):
    """
    Extracts evaluation scores from the AI's markdown response using regex for robustness.
    """
    scores = {}
    section_pattern = re.compile(r"####\s*\d+\.\s*(.*)")
    score_pattern = re.compile(r"\*\*Score:\*\* (\d+)\s*/\s*\d+")
    confidence_pattern = re.compile(r"✅ Confidence Score:\s*(\d+)\s*/\s*\d+")
    current_section = ""
    for line in markdown_text.strip().split("\n"):
        section_match = section_pattern.match(line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        score_match = score_pattern.search(line)
        if score_match and current_section:
            try:
                score = int(score_match.group(1))
                scores[current_section] = score
            except ValueError:
                pass
            finally:
                current_section = ""
        confidence_match = confidence_pattern.search(line)
        if confidence_match:
            try:
                val = int(confidence_match.group(1))
                scores["Bayesian Confidence Score"] = val
            except ValueError:
                pass
    return scores


main_prompt = PromptTemplate(
    input_variables=["query", "user_role", "context"],
    template="""
You are a helpful, knowledgeable, and honest AI healthcare assistant.
You are assisting a user with the role: **{user_role}**.
Use the document context provided below as your primary source.

If the context does not fully answer the question, supplement with basic general clinical knowledge—but indicate that clearly.
Ensure the answer is tailored to the user's role and is easy to understand for them.

Context:
{context}

Question:
{query}

Answer:
"""
)

# --- Original response_eval_prompt restored ---
response_eval_prompt = PromptTemplate(
    input_variables=["question", "context", "response", "user_role"],
    template="""
You are a **healthcare evaluation expert specializing in tuberculosis and diabetes comorbidity**.

Your task is to **assess the quality, reliability, and usefulness** of a response generated by an AI assistant,
which is grounded in the official WHO report titled:
**“Collaborative Framework for Care and Control of Tuberculosis and Diabetes.”**

---

Use the following WHO-derived context:
{context}

User’s Role: {user_role}  
User’s Question:
{question}

AI Assistant’s Response:
{response}

---

### Evaluation Criteria:

1. **Factual Accuracy**: How accurate are the medical facts presented? (1-5)
2. **Medical Relevance**: Is the information directly pertinent to the medical context of the question and the document? (1-5)
3. **Evidence Grounding**: How well is the response supported by the provided document context? (1-5)
4. **Role-Based Clarity**: Is the language and complexity appropriate for the specified user role? (1-5)

Score each 1–5, and provide a short explanation for each.  
At the end, compute a Confidence Score based on the following formula:

✅ Confidence Score =  
(Factual Accuracy × 0.4 + Medical Relevance × 0.25 + Evidence Grounding × 0.25 + Role-Based Clarity × 0.1) × 20

Explain any weight adjustments made based on user role. For instance, for a 'Patient' role, Role-Based Clarity might have a slightly higher implicit emphasis.

---

### Output Format:

#### 1. Factual Accuracy  
**Score:** [Score] / 5  
**Explanation:** [Brief explanation]

#### 2. Medical Relevance  
**Score:** [Score] / 5  
**Explanation:** [Brief explanation]

#### 3. Evidence Grounding  
**Score:** [Score] / 5  
**Explanation:** [Brief explanation]

#### 4. Role-Based Clarity  
**Score:** [Score] / 5  
**Explanation:** [Brief explanation]

---

### ✅ Confidence Score: [Score] / 100  
[Explanation for weight adjustments, e.g., "We emphasized clarity due to patient role."]
"""
)

# --- Original LLM and Chain Initialization restored ---
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.4, google_api_key=api_key)
    llm_chain = LLMChain(llm=llm, prompt=main_prompt)
    response_eval_chain = LLMChain(llm=llm, prompt=response_eval_prompt)
    # sentiment_chain was not in your original code, so it's not included here
except Exception as e:
    st.error(f"Error initializing Google Generative AI models: {e}")
    st.info("Please check your GOOGLE_API_KEY and ensure it has access to the 'gemini-2.5-flash' model.")
    st.stop()

st.set_page_config(page_title="Healthcare RAG Assistant", layout="wide")
st.markdown("<h1 style='text-align: center;'>🩺 Healthcare RAG Assistant</h1>",
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
    "🔍 Ask your question", placeholder="e.g., What lifestyle changes should TB-diabetes patients adopt?")
submit = st.button("Get Answer 💡", use_container_width=True)

if submit:
    if not question.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Generating response..."):
            try:
                if any(kw in question.lower() for kw in ["lifestyle", "diet", "nutrition", "exercise", "behavior", "physical activity", "healthy habits"]):
                    docs_to_use = filter_lifestyle_chunks(chunks)
                    if not docs_to_use:
                        docs_to_use = retriever.invoke(question)
                else:
                    docs_to_use = retriever.invoke(question)
                top_chunk_content = docs_to_use[0].page_content if docs_to_use else ""
                if not top_chunk_content:
                    st.warning(
                        "No highly relevant information found in the document for your question. Providing a general answer based on broad knowledge.")
                    context_for_llm = "No specific information was found in the provided document for this query. The following answer is based on general healthcare knowledge."
                else:
                    context_for_llm = top_chunk_content

                # Generate initial response
                response = llm_chain.invoke({
                    "query": question,
                    "user_role": user_role,
                    "context": context_for_llm
                })
                raw_answer = response.get("text", "").strip()

                # Apply guardrails validation
                is_valid, validation_msg = guardrails.validate_response(
                    raw_answer, user_role)

                if not is_valid:
                    logger.warning(f"Guardrail triggered: {validation_msg}")
                    # Regenerate with safety constraints if validation fails
                    response = llm_chain.invoke({
                        "query": f"[SAFETY MODE] {question}",
                        "user_role": user_role,
                        "context": f"{context_for_llm}\n\nIMPORTANT: Provide only medically validated information in simple terms."
                    })
                    raw_answer = response.get("text", "").strip()
                # Apply simplification (your existing function untouched)
                answer = simplify_for_role(raw_answer, user_role)

                # Rest of your existing evaluation and display logic
                score_response = response_eval_chain.invoke({
                    "question": question,
                    "context": context_for_llm,
                    "response": answer,
                    "user_role": user_role
                })
                eval_scores = extract_eval_scores_markdown(
                    score_response["text"])
                st.session_state["conversations"][user_role].append({
                    "user": question,
                    "ai": answer,
                    "validated": is_valid
                })

                # Your existing display logic
                st.markdown("### 💡 Response")
                for line in answer.split("\n"):
                    if line.strip():
                        st.write("🔹 " + line.strip())

                # Add guardrails validation indicator
                if not is_valid:
                    st.warning(
                        f"⚠️ Response modified by safety guardrails: {validation_msg}")

                st.markdown("### 📈 Evaluation Metrics")
                for metric in ["Factual Accuracy", "Medical Relevance", "Evidence Grounding", "Role-Based Clarity"]:
                    if metric in eval_scores:
                        st.markdown(
                            f"**{metric}:** `{eval_scores[metric]} / 5`")

                    else:
                        st.markdown(f"**{metric}:** `N/A`")

                if "Bayesian Confidence Score" in eval_scores:
                    st.markdown(
                        f"**🧠 Confidence Score:** `{eval_scores['Bayesian Confidence Score']} / 100`")
                else:
                    st.markdown(f"**🧠 Confidence Score:** `N/A`")

            except ResourceExhausted:
                st.error(
                    "Gemini API quota exceeded. Please try again later or check your Google Cloud Console for quota limits.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.info(
                    "Please ensure your GOOGLE_API_KEY is correctly set and has the necessary permissions.")

st.markdown("---")
if st.session_state["conversations"][user_role]:
    st.markdown("### 📜 Conversation History")
    for idx, msg in enumerate(reversed(st.session_state["conversations"][user_role])):
        st.markdown(f"**Me:** {msg['user']}")
        st.markdown(f"**AI:** {msg['ai']}")
        st.markdown("---")

if st.button("Clear Conversation 🧹", use_container_width=True):
    st.session_state["conversations"][user_role] = []
    st.experimental_rerun()
