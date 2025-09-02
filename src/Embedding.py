# %%

def download_emb_model():
    from huggingface_hub import snapshot_download

    # 下载 BAAI 官方的 BGE-Small 中文模型（自带 sentence_bert_config.json）
    snapshot_download(
        repo_id="BAAI/bge-small-zh", 
        local_dir="../model/bge-Small",
        local_dir_use_symlinks=False,  # Windows 必加
        allow_patterns=["*.json", "*.bin", "*.txt", "*.model"] 
    )

# download_emb_model()

# %%
import os

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import  FAISS
from langchain_chroma import Chroma
from LoadData import load_document,chunk_data

from tool import skip_execution
IS_SKIP =False

# %% [markdown]
# https://huggingface.co/BAAI/bge-small-zh

# %%

def get_embedding(embedding_name):
    """
    根据embedding名称加载对应的嵌入模型
    """
    # 通用模型参数配置
    model_kwargs = {'device': 'cuda'}  
    encode_kwargs = {'normalize_embeddings': True}  # 归一化嵌入向量
    
    embedding_path = os.path.join('..',"model",embedding_name)
    print(embedding_path)
    
    return HuggingFaceEmbeddings(
            model_name=embedding_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    




# %% [markdown]
# * 闭源 API 模型	: OpenAIEmbeddings
#   * OpenAI Ada-002、Anthropic Claude		
# * 开源本地模型	: HuggingFaceEmbeddings
#   * BERT、Sentence-BERT（如 all-MiniLM）		
# * 云厂商模型	: AliyunEmbeddings
#   * 阿里云通义千问嵌入、腾讯云向量嵌入	 

# %% [markdown]
# 使用embedding模型持久化存储，目前常用的中文模型是bge-large-zh-v1.5

# %%
@skip_execution(IS_SKIP)
def test_emb(): 
    embedding=get_embedding("bge-Small")
          # 测试生成嵌入向量
    test_text = "这是一个测试句子，用于验证嵌入模型是否正常工作。"
    embedding_vector = embedding.embed_query(test_text)
        
        # 输出结果信息
    print(f"嵌入向量维度: {len(embedding_vector)}")
    print(f"嵌入向量前5个值: {embedding_vector[:5]}")
 
    
test_emb()

# %%
# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings_chroma(embedding_name, chunks, persist_dir="../db/chroma_db"):
    """
    创建并保存 Chroma 向量库
    """
    # 获取嵌入模型
    embeddings = get_embedding(embedding_name)
    
    # 创建向量库时指定保存路径
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir  # 指定本地保存目录
    )
    
    # 打印保存信息
    print(f"Chroma 向量库已保存到: {os.path.abspath(persist_dir)}")
    return vector_store

def load_embeddings_chroma(embedding_name, persist_dir):
    """
    加载已保存的 Chroma 向量库
    """
    # 获取与创建时相同的嵌入模型（必须一致，否则向量不兼容）
    embeddings = get_embedding(embedding_name)
    
    # 加载本地向量库
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    print(f"Chroma 向量库已从 {os.path.abspath(persist_dir)} 加载")
    return vector_store


# %%
@skip_execution(IS_SKIP)
def test_chroma():
    path = "../datasets/tangshi.pdf"
    vector_path ="../db/chroma_db"
    embedding_name="bge-Small"
    data = load_document(path)
    chunks = chunk_data(data) 
    create_embeddings_chroma(embedding_name,chunks,vector_path)
    load_embeddings_chroma(embedding_name,vector_path)
test_chroma()

# %% [markdown]
# ### Faiss
# Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. 
# 
# https://faiss.ai/
# 
# https://github.com/facebookresearch/faiss?tab=readme-ov-file

# %%

def create_embeddings_faiss( embedding_name, chunks,vector_db_path="../db/faiss_db"):
    """
    使用FAISS向量数据库，并保存
    """
    embeddings = get_embedding(embedding_name)
    db = FAISS.from_documents(chunks, embeddings)

    if not os.path.isdir(vector_db_path):
        os.mkdir(vector_db_path)

    db.save_local(folder_path=vector_db_path)
    return db


def load_embeddings_faiss( embedding_name,vector_db_path):
    """
    加载向量库
    """
    embeddings = get_embedding(embedding_name)
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db

# %%
@skip_execution(IS_SKIP)
def test_faiss():
    path = "../datasets/tangshi.pdf"
    vector_path ="../db/faiss_db"
    embedding_name="bge-Small"
    data = load_document(path)
    chunks = chunk_data(data) 
    create_embeddings_faiss(embedding_name,chunks,vector_path)
    load_embeddings_faiss(embedding_name,vector_path)
    
test_faiss()


