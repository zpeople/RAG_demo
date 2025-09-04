#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
try:
    get_ipython
    current_dir = os.getcwd()
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Set path，temporary path expansion
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import  FAISS
from langchain_chroma import Chroma
from LoadData import load_document,chunk_data,load_json

from tool import skip_execution
IS_SKIP =False

embedding_name="BAAI/bge-small-zh"


# In[ ]:


def download_emb_model(name):
    from huggingface_hub import snapshot_download

    # 下载 BAAI 官方的 BGE-Small 中文模型（自带 sentence_bert_config.json）
    snapshot_download(
        repo_id=name, 
        local_dir=os.path.join(project_dir,"model",name),
        local_dir_use_symlinks=False,  # Windows 必加
        allow_patterns=["*.json", "*.bin", "*.txt", "*.model"] 
    )

# download_emb_model(embedding_name)


# https://huggingface.co/BAAI/bge-small-zh

# In[ ]:


def get_embedding(embedding_name):
    """
    根据embedding名称加载对应的嵌入模型
    """
    # 通用模型参数配置
    model_kwargs = {'device': 'cuda'}  
    encode_kwargs = {'normalize_embeddings': True}  # 归一化嵌入向量
    
    embedding_path = os.path.join(project_dir,"model",embedding_name)
    print(embedding_path)
    
    return HuggingFaceEmbeddings(
            model_name=embedding_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    


# * 闭源 API 模型	: OpenAIEmbeddings
#   * OpenAI Ada-002、Anthropic Claude		
# * 开源本地模型	: HuggingFaceEmbeddings
#   * BERT、Sentence-BERT（如 all-MiniLM）		
# * 云厂商模型	: AliyunEmbeddings
#   * 阿里云通义千问嵌入、腾讯云向量嵌入	 

# 使用embedding模型持久化存储，目前常用的中文模型是bge-large-zh-v1.5

# In[ ]:


@skip_execution(IS_SKIP)
def test_emb(): 
    embedding=get_embedding(embedding_name)
          # 测试生成嵌入向量
    test_text = "这是一个测试句子，用于验证嵌入模型是否正常工作。"
    embedding_vector = embedding.embed_query(test_text)
        
        # 输出结果信息
    print(f"嵌入向量维度: {len(embedding_vector)}")
    print(f"嵌入向量前5个值: {embedding_vector[:5]}")
 
    
test_emb()


# ### Chroma
# 
# 优势是以最小成本实现向量数据库的核心价值—— 无需关注底层细节，快速搭建可用的向量检索系统，尤其适合原型开发、中小规模应用或对部署复杂度敏感的场景。

# In[ ]:


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings_chroma1(embedding_name, chunks, persist_dir=os.path.join(project_dir,"db/chroma_db")):
    """
    创建并保存 Chroma 向量库
    """
    # 获取嵌入模型
    embeddings = get_embedding(embedding_name)
    if not os.path.isdir(persist_dir):
        os.mkdir(persist_dir)

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


# In[ ]:


'''
新版取消了vector_store.persist()
'''
# def create_embeddings_chroma(embedding_name, chunks, persist_dir=os.path.join(project_dir, "db/chroma_db"), batch_size=1000):
#     embeddings = get_embedding(embedding_name)
#     os.makedirs(persist_dir, exist_ok=True)
    
#     # 初始化Chroma（如果目录已存在则加载，实现断点续传）
#     if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
#         if vector_store ==None: 
#             vector_store = Chroma(
#                 persist_directory=persist_dir,
#                 embedding_function=embeddings
#             )
#         # 获取已存在的文档数量（用于断点续传）
#         existing_count = vector_store._collection.count()
#         print(f"检测到已有向量库，已包含 {existing_count} 条文档")
#     else:
#         # 新建向量库
#         initial_batch = chunks[:batch_size]
#         if not initial_batch:
#             raise ValueError("❌ chunks 为空，无法初始化 Chroma 向量库")

#         vector_store = Chroma.from_documents(
#             documents=initial_batch,
#             embedding=embeddings,
#             # persist_directory=persist_dir  # 这儿有bug
#         )
   
#         existing_count = len(initial_batch)
       
#     # 边界检查：避免索引越界
#     if existing_count >= len(chunks):
#         print(" 所有 chunks 已处理，无需更新 FAISS 向量库")
#         return vector_store
#     # 从断点开始处理剩余文档
#     remaining_chunks = chunks[existing_count:]

#     print(len(remaining_chunks))
#     total_batches = (len(remaining_chunks) + batch_size - 1) // batch_size
    
#     for i in range(total_batches):
#         start = i * batch_size
#         end = start + batch_size
#         batch = remaining_chunks[start:end]
        
#         # 增量添加批次
#         vector_store.add_documents(batch)
        
#         # 每10批保存一次（减少IO次数）
#         if (i + 1) % 10 == 0:
#             # vector_store.persist()
#             print(f"已处理 {start + end} 条文档，进度：{((i + 1) / total_batches) * 100:.2f}%")
    
 
#     print(f"Chroma 向量库已保存到: {os.path.abspath(persist_dir)}，共 {vector_store._collection.count()} 条文档")
#     return vector_store


# In[ ]:


@skip_execution(IS_SKIP)
def test_chroma():
    path = os.path.join(project_dir,"datasets/tangshi.pdf") 
    vector_path =os.path.join(project_dir,"db/chroma_db") 

    data = load_document(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    print(len(chunks))
    create_embeddings_chroma(embedding_name,chunks,vector_path)
    # load_embeddings_chroma(embedding_name,vector_path)
test_chroma()


# ### Faiss
# Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. 
# 
# 向量搜索算法库（纯工具库），专注于高维向量的高效相似性搜索，核心价值是提供优化的索引算法（如 IVF、HNSW、PQ 等），解决 “如何快速从海量向量中找到相似结果” 的技术问题
# 
# https://faiss.ai/
# 
# https://github.com/facebookresearch/faiss?tab=readme-ov-file
# 
# * 暴力搜索（Brute-force）：
# 索引类型：IndexFlatL2（L2 距离）、IndexFlatIP（内积，可用于余弦相似度）。
# 特点：精确但速度慢，适合小规模数据（万级以下）。
# * 倒排文件索引（IVF）：
# 索引类型：IndexIVFFlat、IndexIVFPQ等。
# 特点：将向量聚类到多个桶（cluster），搜索时仅在目标桶内进行，平衡速度和精度，适合百万到亿级数据。
# * 分层导航小世界网络（HNSW）：
# 索引类型：IndexHNSWFlat。
# 特点：基于图结构的近似搜索，速度快、精度高，适合高维向量和实时场景。
# * 乘积量化（PQ）：
# 索引类型：IndexPQ、IndexIVFPQ等。
# 特点：将向量分段并量化，大幅降低内存占用，适合超大规模数据（十亿级）。

# In[ ]:


def create_embeddings_faiss( embedding_name, chunks,persist_dir=os.path.join(project_dir,"db/faiss_db") ):
    """
    使用FAISS向量数据库，并保存
    """
    embeddings = get_embedding(embedding_name)
    db = FAISS.from_documents(chunks, embeddings)

    if not os.path.isdir(persist_dir):
        os.mkdir(persist_dir)

    db.save_local(folder_path=persist_dir)
    return db


def load_embeddings_faiss( embedding_name,persist_dir):
    """
    加载向量库
    """
    embeddings = get_embedding(embedding_name)
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return db


# In[15]:


import math
import os
from langchain.vectorstores import FAISS

def create_embeddings_faiss(
    embedding_name,
    chunks,
    persist_dir=os.path.join(project_dir, "db/faiss_db"),
    batch_size=1000
):
    embeddings = get_embedding(embedding_name)
    os.makedirs(persist_dir, exist_ok=True)

    faiss_index_path = os.path.join(persist_dir, "index.faiss")
    faiss_docstore_path = os.path.join(persist_dir, "index.pkl")

    # 初始化 FAISS 向量库
    if os.path.exists(faiss_index_path) and os.path.exists(faiss_docstore_path):
        print("检测到已有 FAISS 向量库，尝试加载...")
        db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        existing_count = len(db.docstore._dict)
        print(f"已加载 {existing_count} 条文档")
    else:
        print("创建新的 FAISS 向量库...")
        initial_batch = chunks[:batch_size]
        if not initial_batch:
            raise ValueError(" chunks 为空，无法初始化 FAISS 向量库")
        db = FAISS.from_documents(initial_batch, embeddings)
        existing_count = len(initial_batch)

    # 边界检查：避免索引越界
    if existing_count >= len(chunks):
        print(" 所有 chunks 已处理，无需更新 FAISS 向量库")
        return db

    remaining_chunks = chunks[existing_count:]
    total_batches = math.ceil(len(remaining_chunks) / batch_size)

    for i in range(total_batches):
        start = i * batch_size
        end = min(start + batch_size, len(remaining_chunks))
        batch = remaining_chunks[start:end]

        if not batch:
            print(f"第 {i+1} 批为空，跳过")
            continue

        print(f"正在处理第 {i+1}/{total_batches} 批（{len(batch)} 条）")

        db.add_documents(batch)

        # 每10批保存一次
        if (i + 1) % 10 == 0 or i == total_batches - 1:
            db.save_local(persist_dir)
            print(f"已保存至 {persist_dir}，当前总文档数：{len(db.docstore._dict)}")

    print(f"FAISS 向量库构建完成，共 {len(db.docstore._dict)} 条文档")
    return db


# In[ ]:


@skip_execution(IS_SKIP)
def test_faiss():
    path = os.path.join(project_dir,"datasets/tangshi.pdf") 
    vector_path =os.path.join(project_dir,"db/faiss_db") 
    data = load_document(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    create_embeddings_faiss(embedding_name,chunks,vector_path,100)
    load_embeddings_faiss(embedding_name,vector_path)
    
test_faiss()


# In[ ]:


@skip_execution(IS_SKIP)
def test_law():
    path = os.path.join(project_dir,"datasets","chinese_law_ft_dataset.json") 
    vector_path =os.path.join(project_dir,"db/law_db") 
    data = load_json(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    create_embeddings_faiss(embedding_name,chunks,vector_path,batch_size=100)
    # load_embeddings_chroma(embedding_name,vector_path)
    
test_law()

