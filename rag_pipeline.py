import os

from typing import List,  Tuple


# LangChain Imports - UPDATED TO USE LCEL
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
# New imports for LCEL RAG components
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate  # Using ChatPromptTemplate is standard for modern LLMs
from langchain_core.runnables import Runnable
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Configuration ---
# You need to ensure these environment variables are set up for Azure LLM
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Configuration for the Hugging Face Embeddings Model
# Using a popular and efficient Sentence Transformer model
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

endpoint = "https://dev-openai-service-01.openai.azure.com/"
model_name = "gpt-4o"
deployment = "b2grp4-e51444c1-62b6-4934-a875-d7d23fe25e53"
subscription_key = os.getenv('my_key_value')
api_version = "2024-12-01-preview"

# Initialize Azure OpenAI Components (LLM remains the same)
llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        openai_api_key=subscription_key,
   )

# Initialize Hugging Face Embeddings
# The model will be downloaded automatically the first time it's run.
# Consider changing 'cpu' to 'cuda' if you have a GPU for faster embedding generation.
embeddings = HuggingFaceEmbeddings(
    model_name=HF_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

# --- RAG Setup with LCEL (Unchanged, as it uses the 'embeddings' object) ---

def setup_rag_pipeline(documents: List[Document]) -> Tuple[Chroma, Runnable]:
    # ... (function body remains the same)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\nclass", "\n// File:", "\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,  # Uses the HuggingFaceEmbeddings object
        collection_name="java_code_documentation"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 180})

    prompt_template_string = """
As an expert technical writer and Python programmer specializing in Java, generate a complete, professional, and well-structured technical documentation suite for all Java files and code chunks provided in the upcoming CODE CONTEXT.
The documentation must achieve comprehensive coverage, detailing every component, and include the following sections:

1. Project Overview and Architecture
Full Project Overview: A high-level description of the entire project's scope, primary goal, and domain.
Full Architecture Summary: An overview of the system's structure, how the components interact, and the primary design patterns used.

2. Component Documentation (Classes and Methods)
For every single Java class, inner class, and static class found within the CODE CONTEXT:

- Purpose and Role: A clear explanation of the class's responsibility within the project.
- All Methods and Behavior: A detailed explanation of all public, protected, and private methods, including their parameters, return values, side effects, and precise operational logic.
- Relationships / Dependencies: Identify and explain all dependencies (other classes, interfaces, external libraries) and how this class interacts with them.
- Example Usage: If the CODE CONTEXT contains usage examples (e.g., a main method, unit tests, or clear initialization code), document and explain it. If not, construct a simple, illustrative usage example.

3. Notes and Observations
Code Observations: Any notable design choices, potential limitations, performance considerations, or areas for future enhancement.

3. Test cases 
- Generate the documentation for creating the test cases for various classes found within the CODE CONTEXT:

Your primary goal is exhaustive, technical accuracy, and clarity. Ensure no class, or method is omitted from the final documentation. Ensure all the details provided so that using this documentation expert developer should be able to develop the Spring boot application

------ CODE CONTEXT ------
    {context}
    
    Write high-quality Markdown documentation.
    """

    prompt = ChatPromptTemplate.from_template(prompt_template_string)

    document_chain = create_stuff_documents_chain(llm, prompt)

    qa_chain = create_retrieval_chain(retriever, document_chain)

    return vectorstore, qa_chain

