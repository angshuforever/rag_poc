import os
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

# Initialize Sentence Transformers for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"


class SentenceTransformerEmbedding(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return embedding_model.encode(texts).tolist()


# Initialize collection with the custom embedding function
embedding_func = SentenceTransformerEmbedding()
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_func
)


def search_web(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo and extract content from webpages.
    """
    results = []

    try:
        with DDGS() as ddgs:
            search_results = [r for r in ddgs.text(query, max_results=num_results)]

        for result in search_results:
            try:
                # Get webpage content
                response = requests.get(result['link'], timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Extract text
                    text = soup.get_text(separator=' ', strip=True)

                    # Clean and format text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    results.append({
                        'title': result['title'],
                        'link': result['link'],
                        'content': text[:1000]  # Limit content length
                    })
            except Exception as e:
                print(f"Error processing {result['link']}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error during web search: {str(e)}")

    return results


def add_web_content_to_collection(web_results: List[Dict[str, str]]):
    """
    Add web search results to the vector database.
    """
    for i, result in enumerate(web_results):
        doc_id = f"web_content_{i}"
        embedding = embedding_model.encode(result['content']).tolist()

        collection.upsert(
            ids=[doc_id],
            documents=[result['content']],
            embeddings=[embedding],
            metadatas=[{"source": result['link'], "title": result['title']}]
        )


def query_documents(question: str, n_results: int = 2, similarity_threshold: float = 0.7):
    """
    Query documents with a similarity threshold and web search fallback.
    """
    # First, query the existing collection
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    relevant_chunks = results["documents"][0]
    distances = results.get("distances", [[]])[0]

    # Check if we have good matches (using distance/similarity threshold)
    good_matches = [chunk for chunk, dist in zip(relevant_chunks, distances)
                    if dist <= similarity_threshold]

    if not good_matches:
        print("No relevant information found in database, searching web...")
        web_results = search_web(question)

        if web_results:
            # Add web content to the collection
            add_web_content_to_collection(web_results)

            # Query again including the new content
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
            relevant_chunks = results["documents"][0]

    return relevant_chunks


def generate_response(question: str, relevant_chunks: List[str]) -> str:
    context = "\n\n".join(relevant_chunks)
    prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of "
            "retrieved context to answer the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message


if __name__ == "__main__":
    # Example usage
    question = "tell me about elon musk"
    relevant_chunks = query_documents(question, similarity_threshold=0.7)
    answer = generate_response(question, relevant_chunks)
    print(answer.content)