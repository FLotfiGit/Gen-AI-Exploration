import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.retrievers.multi_query_retriever import MultiQueryRetriever
from langchain_openai import ChatOpenAI
import torch

load_dotenv()

# vector store
cur_dir = os.getcwd()
vdb_dir = os.path.join(cur_dir, 'db', 'chroma_db')
rag_vdb_dir = os.path.join(vdb_dir, 'rag_articles')

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Define 'chunks' as a list of documents to embed
from langchain.docstore.document import Document

chunks = [
    Document(page_content="Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with generative models.", metadata={"origin_url": "https://example.com/rag_intro"}),
    Document(page_content="RAG is needed to provide up-to-date and factual information to language models by retrieving external knowledge.", metadata={"origin_url": "https://example.com/rag_need"}),
    Document(page_content="RAG works by retrieving relevant documents and then using them as context for a generative model to answer questions.", metadata={"origin_url": "https://example.com/rag_work"}),
]

if not os.path.exists(rag_vdb_dir):
    db = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=rag_vdb_dir
    )
else:
    print("Vector store already exists for rag topic. No need to initialize.")

db = Chroma(persist_directory=rag_vdb_dir, embedding_function=embeddings_model)

# Retriever. ~~~~~~~~~
query = "what is RAG? why do we need RAG? how does it work?"
llm = ChatOpenAI(temperature=0)
retriever_multi_query = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(
        search_type="mmr",search_kwargs={"k": 5}
    ), llm=llm
)
relevant_chunks = retriever_multi_query.invoke(query)


# Generation ~~~~~~~
combined_input = (
    "Here is some context and source url that might help answer the question: "
    + query
    + "\n\nRelevant context(which starts with ### and ends with ###) and source url:\n"
    + "\n\n".join([f"source_url:{chunk.metadata['origin_url']}\ncontent:###\n{chunk.page_content}\n###" for chunk in relevant_chunks])
    + "\n\nPlease provide an answer based only on the provided context. If the answer is not found in it, respond with 'I'm not sure'."
    + " When providing an answer, include citations from the provided context to increase credibility."
    + "\n\nAfter the answer, give three relative questions attractive to the user and output one by one in the format 'Relative question {#no}: {relative question}'."
)


# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o").to("cuda" if torch.cuda.is_available() else "cpu")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]
# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- User Request ---")
print(query)
print("\n--- Generated Response ---")
print(result.content)

def extract_source_urls(chunks):
    """Extract all source URLs from a list of Document chunks."""
    return [chunk.metadata.get('origin_url') for chunk in chunks if 'origin_url' in chunk.metadata]

def extract_page_contents(chunks):
    """Extract all page contents from a list of Document chunks."""
    return [chunk.page_content for chunk in chunks]