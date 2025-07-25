import asyncio
import time
from typing import Any

import PyPDF2
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Global rate limiting
last_api_call_time = 0
MIN_API_INTERVAL = 2.0  # Minimum 2 seconds between API calls

def rate_limit():
    """Ensure minimum interval between API calls"""
    global last_api_call_time
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    
    if time_since_last_call < MIN_API_INTERVAL:
        sleep_time = MIN_API_INTERVAL - time_since_last_call
        print(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    last_api_call_time = time.time()

@retry(wait=wait_random_exponential(multiplier=2, max=300), stop=stop_after_attempt(5))
def get_embeddings(
    embedding_client: Any, embedding_model: str, text: str, output_dim: int = 768
) -> list[float]:
    """
    Generate embeddings for text with retry logic and rate limiting for API quota management.
    """
    rate_limit()  # Apply rate limiting
    
    try:
        response = embedding_client.models.embed_content(
            model=embedding_model,
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=output_dim),
        )
        return [response.embeddings[0].values]
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"Quota exhausted for embeddings. Waiting longer before retry...")
            time.sleep(60)  # Wait 1 minute for quota reset
            return None
        print(f"Error generating embeddings: {str(e)}")
        raise


def build_index(
    document_paths: list[str],
    embedding_client: Any,
    embedding_model: str,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """
    Build searchable index from a list of PDF documents with page-wise processing.
    """
    all_chunks = []

    for doc_path in document_paths:
        try:
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    chunks = [
                        page_text[i : i + chunk_size]
                        for i in range(0, len(page_text), chunk_size)
                    ]

                    for chunk_num, chunk_text in enumerate(chunks):
                        embeddings = get_embeddings(
                            embedding_client, embedding_model, chunk_text
                        )

                        if embeddings is None:
                            print(
                                f"Warning: Could not generate embeddings for chunk {chunk_num} on page {page_num + 1}"
                            )
                            continue

                        chunk_info = {
                            "document_name": doc_path,
                            "page_number": page_num + 1,
                            "page_text": page_text,
                            "chunk_number": chunk_num,
                            "chunk_text": chunk_text,
                            "embeddings": embeddings,
                        }
                        all_chunks.append(chunk_info)

        except Exception as e:
            print(f"Error processing document {doc_path}: {str(e)}")
            continue

    if not all_chunks:
        raise ValueError("No chunks were created from the documents")

    return pd.DataFrame(all_chunks)


def get_relevant_chunks(
    query: str,
    vector_db: pd.DataFrame,
    embedding_client: Any,
    embedding_model: str,
    top_k: int = 3,
) -> str:
    """
    Retrieve the most relevant document chunks for a query using similarity search.
    """
    try:
        query_embedding = get_embeddings(embedding_client, embedding_model, query)

        if query_embedding is None:
            return "Could not process query due to quota issues"

        similarities = [
            cosine_similarity(query_embedding, chunk_emb)[0][0]
            for chunk_emb in vector_db["embeddings"]
        ]

        top_indices = np.argsort(similarities)[-top_k:]
        relevant_chunks = vector_db.iloc[top_indices]

        context = []
        for _, row in relevant_chunks.iterrows():
            context.append(
                {
                    "document_name": row["document_name"],
                    "page_number": row["page_number"],
                    "chunk_number": row["chunk_number"],
                    "chunk_text": row["chunk_text"],
                }
            )

        return "\n\n".join(
            [
                f"[Page {chunk['page_number']}, Chunk {chunk['chunk_number']}]: {chunk['chunk_text']}"
                for chunk in context
            ]
        )

    except Exception as e:
        print(f"Error getting relevant chunks: {str(e)}")
        return "Error retrieving relevant chunks"


@retry(wait=wait_random_exponential(multiplier=2, max=300), stop=stop_after_attempt(5))
async def generate_answer(
    query: str, context: str, llm_client: Any, llm_model: str, modality: str = "text"
) -> str:
    """
    Generate answer using LLM with retry logic and rate limiting for API quota management.
    """
    try:
        if context in [
            "Could not process query due to quota issues",
            "Error retrieving relevant chunks",
        ]:
            return "Can't Process, Quota Issues"

        prompt = f"""Based on the following context, please answer the question.

        Context:
        {context}

        Question: {query}

        Answer:"""

        # Add rate limiting before making API calls
        await asyncio.sleep(2)  # Async rate limiting
        
        if modality == "text":
            response = await generate_text_content(prompt, llm_client, llm_model)
            return response

        elif modality == "audio":
            return await generate_audio_content(prompt, llm_client, llm_model)

    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"Quota exhausted for generation. Waiting longer before retry...")
            await asyncio.sleep(120)  # Wait 2 minutes for quota reset
            return "Can't Process, Quota Issues - Please try again later"
        print(f"Error generating answer: {str(e)}")
        return "Error generating answer"

async def generate_text_content(query: str, client: Any, model: str) -> str:
    """Function to generate text content using Gemini live API with rate limiting."""
    try:
        config = types.LiveConnectConfig(response_modalities=["TEXT"])

        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send(input=query, end_of_turn=True)
            response = []
            async for message in session.receive():
                try:
                    if message.text:
                        response.append(message.text)
                except AttributeError:
                    pass
                if message.server_content.turn_complete:
                    return "".join(str(x) for x in response)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"Text generation quota exhausted: {str(e)}")
            return "Quota exhausted - please try again later"
        raise


async def generate_audio_content(query: str, client: Any, model: str):
    """Function to generate audio response with quota handling."""
    try:
        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
                "language_code": "en-US"  # Changed from vi-VI to en-US for better compatibility
            },
        }

        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send(input=query, end_of_turn=True)
            audio_parts = []
            async for message in session.receive():
                if message.server_content.model_turn:
                    for part in message.server_content.model_turn.parts:
                        if part.inline_data:
                            audio_parts.append(
                                np.frombuffer(part.inline_data.data, dtype=np.int16)
                            )
                if message.server_content.turn_complete:
                    if audio_parts:
                        audio_data = np.concatenate(audio_parts, axis=0)
                        return audio_data
                    break
        return None
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"Audio generation quota exhausted: {str(e)}")
            return None
        raise


async def rag(
    question: str,
    vector_db: pd.DataFrame,
    embedding_client: Any,
    embedding_model: str,
    llm_client: Any,
    top_k: int,
    llm_model: str,
    modality: str = "text",
) -> Any:
    """
    RAG Pipeline.
    """
    try:
        relevant_context = get_relevant_chunks(
            question, vector_db, embedding_client, embedding_model, top_k=top_k
        )

        return await generate_answer(
            question,
            relevant_context,
            llm_client,
            llm_model,
            modality=modality,
        )

    except Exception as e:
        print(f"Error processing question '{question}': {str(e)}")
        return {"question": question, "generated_answer": "Error processing question"} 