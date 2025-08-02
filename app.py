import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
#from dotenv import load_dotenv
import gradio as gr

# Load API Key from .env file
#load_dotenv()
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in Hugging Face secrets.")

#if not openai_api_key:
 #   raise Exception("Please set your OPENAI_API_KEY in the .env file")

# STEP 1: Load and Split Document
print("🔹 Loading document...")
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

print("🔹 Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

# STEP 2: Embed and Store in FAISS
print("🔹 Creating embeddings...")
embedding_model = OpenAIEmbeddings()

print("🔹 Creating vector store...")
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# STEP 3: LLM and QA Chain
print("🔹 Initializing LLM...")
llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

print("🔹 Creating QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# STEP 4: Define Function for Querying
def ask_bot(query):
    if not query.strip():
        return "Please enter a question."
    response = qa_chain.run(query)
    return response

# STEP 5: Optional Gradio UI
print("✅ Ready! Launching UI...")
gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs="text",
    title="Ask Me Anything – AI Chatbot",
    description="Ask questions based on your resume or other uploaded document."
).launch()
