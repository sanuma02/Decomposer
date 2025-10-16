import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
#from llama_index.vector_stores import FAISSVectorStore
#from llama_index.llms import OpenAI
#from llama_index.core.llms import OpenAI
from llama_index.llms.openai import OpenAI
#from llama_index.storage.storage_context import StorageContext
from llama_index.core import StorageContext, load_index_from_storage
#from llama_index import load_index_from_storage
import faiss

load_dotenv()

app = FastAPI()


# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine()
# response = query_engine.query("Some question about the data should go here")
# print(response)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables")

# Load documents and build index once on startup
print("Loading documents and building index...")

# You can put text or PDF files inside a folder named 'data'
documents = SimpleDirectoryReader("data").load_data()

# Set up LLM with OpenAI
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

# Create FAISS vector store
#vector_store = FAISSVectorStore(faiss_index=faiss.IndexFlatL2(1536))

#service_context = ServiceContext.from_defaults(llm=llm)

# Build the index
#index = VectorStoreIndex.from_documents(documents, service_context=service_context, vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents)

# Save index to disk (optional)

storage_context = StorageContext.from_defaults(persist_dir="storage")
#index.storage_context.persist(persist_dir="./index_storage")
query_engine = index.as_query_engine()

print("Index built!")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_index(req: QueryRequest):
    try:
        response = query_engine.query(req.query)
        return {"answer": response.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
