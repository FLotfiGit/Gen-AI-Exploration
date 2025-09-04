import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

load_dotenv()

# vector store
cur_dir = os.getcwd()
vdb_dir = os.path.join(cur_dir, 'db', 'chroma_db')
rag_vdb_dir = os.path.join(vdb_dir, 'rag_articles')

if not os.path.exists(rag_vdb_dir):
    db = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=rag_vdb_dir
    )
else:
    print("Vector store already exists for rag topic. No need to initialize.")

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=rag_vdb_dir, embedding_function=embeddings_model)

# Retriever
query = "what is RAG? why do we need RAG? how does it work?"
llm = ChatOpenAI(temperature=0)
retriever_multi_query = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(
        search_type="mmr",search_kwargs={"k": 5}
    ), llm=llm
)
relevant_chunks = retriever_multi_query.invoke(query)


# Generation
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
model = ChatOpenAI(model="gpt-4o")

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