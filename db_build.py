# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.llms import OpenAI
from src.openaikeys import OPENAI_API

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build(local=True, path=cfg.DATA_PATH,vec_path=cfg.DB_FAISS_PATH, file_path=None):

    if os.path.exists(f"vectorstore/{vec_path}"):
        # Load new documents
        new_loader = PyPDFLoader(file_path)
        new_documents = new_loader.load()

        # Split texts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
        documents_split = text_splitter.split_documents(new_documents)

        # Initialize HuggingFaceEmbeddings with the model
        if local:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                               model_kwargs={'device': 'cpu'})
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API)

        f = FAISS.load_local(f"vectorstore/{vec_path}", embeddings)

        f.add_documents(documents_split)

        f.save_local(f"vectorstore/{vec_path}")
    else:
        loader = DirectoryLoader(f"data/{path}",
                                 glob='*.pdf',
                                 loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                       chunk_overlap=cfg.CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)

        if local:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                               model_kwargs={'device': 'cpu'})
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API)

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(f"vectorstore/{vec_path}")



if __name__ == "__main__":
    pass
