import os
from typing import List, Dict, Tuple
from tqdm import tqdm

from util.utility import split_java_by_structure
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.runnables import Runnable
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

import streamlit as st
import re,json

from langchain_core.messages import HumanMessage, SystemMessage

import time
#from tenacity import retry, stop_after_attempt, wait_exponential

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
        request_timeout=120
   )


class DocumentationEvaluation(BaseModel):
    completeness_score: int = Field(description="Score 0-100 for completeness")
    accuracy_score: int = Field(description="Score 0-100 for accuracy")
    clarity_score: int = Field(description="Score 0-100 for clarity")
    actionability_score: int = Field(description="Score 0-100 for actionability")
    detail_score: int = Field(description="Score 0-100 for detail level")
    overall_score: int = Field(description="Overall score 0-100")
    missing_areas: List[str] = Field(description="List of missing areas")
    manual_review_needed: List[str] = Field(description="Items needing manual review")
    improvement_suggestions: List[str] = Field(description="Improvement suggestions")


def create_documents(java_files: Dict[str, str]) -> List[Document]:
    # ... (function body remains the same)
    #documents = []
    docs: List[Document] = []

    # 2. instantiate a language-aware splitter for Java
    splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA,
            chunk_size=1500,
            chunk_overlap=150,
    )

    for path, content in tqdm(java_files.items(), desc="Processing java files"):
            java_text  = f"// File: {path}\n\n{content}"
            # structural split using hybrid splitter
            blocks = split_java_by_structure(java_text)

            # For each structural block, further chunk using the language-aware splitter
            for idx, block in enumerate(blocks):
                block_chunks = splitter.split_text(block)
                for i, ch in enumerate(block_chunks):
                    metadata = {
                        "source": path,
                        "block_index": idx,
                        "chunk_index": i,
                        # optionally include a short preview (first 120 chars)
                        "preview": ch[:120].replace("\n", " ") + ("..." if len(ch) > 120 else ""),
                    }
                    docs.append(Document(page_content=ch, metadata=metadata))

    print(f"Total chunks to embed: {len(docs)}")

    return docs

# --- Documentation Generation and Enhancement (Unchanged) ---

def generate_documentation(qa_chain: Runnable) -> str:
    # ... (function body remains the same)
    query = "Generate the complete, single-file, comprehensive documentation for the entire codebase according to the instructions in the system prompt."

    result = qa_chain.invoke({"input": query},temperature=0.0)

    return result['answer']


def evaluate_documentation( documentation: str) -> Dict:
        parser = PydanticOutputParser(pydantic_object=DocumentationEvaluation)

        system_prompt = """You are an expert code documentation reviewer. Evaluate the quality and completeness of technical documentation.

Assess the documentation on these criteria:
1. Completeness (0-100): Does it cover all major components?
2. Accuracy (0-100): Is the information technically correct?
3. Clarity (0-100): Is it well-written and easy to understand?
4. Actionability (0-100): Can a developer use this for migration?
5. Detail Level (0-100): Is there sufficient technical detail?

Also identify:
- Missing components or areas that need manual review
- Potential inaccuracies or unclear sections
- Suggestions for improvement"""

        user_prompt = f"""Evaluate this documentation against the codebase structure:

DOCUMENTATION:
{documentation[:3000]}

{parser.get_format_instructions()}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = llm.invoke(messages)
            evaluation = parser.parse(response.content)
            return evaluation.model_dump()
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate documentation: {str(e)}. Please ensure Azure OpenAI is configured correctly.")



def refine_documentation( documentation: str, evaluation: Dict) -> str:
        system_prompt = """You are an expert technical documentation writer. Refine and improve documentation based on evaluation feedback."""

        user_prompt = f"""Refine the following documentation based on this evaluation:

ORIGINAL DOCUMENTATION:
{documentation}

EVALUATION FEEDBACK:
Missing Areas: {', '.join(evaluation.get('missing_areas', []))}
Manual Review Needed: {', '.join(evaluation.get('manual_review_needed', []))}
Improvement Suggestions: {', '.join(evaluation.get('improvement_suggestions', []))}

Please provide an improved version that addresses these concerns."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        return response.content
