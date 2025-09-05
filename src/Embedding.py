#!/usr/bin/env python
# coding: utf-8

# In[30]:


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
from langchain_chroma import Chroma
from langchain_community.vectorstores import  FAISS
from langchain_milvus import Milvus
from LoadData import load_document,chunk_data,load_json
import math
from tool import skip_execution
IS_SKIP =True
embedding_name="BAAI/bge-small-zh"


# ## Embedding Model
# ### Download Embedding Model

# * 闭源 API 模型	: OpenAIEmbeddings
#   * OpenAI Ada-002、Anthropic Claude		
# * 开源本地模型	: HuggingFaceEmbeddings
#   * BERT、Sentence-BERT（如 all-MiniLM）		
# * 云厂商模型	: AliyunEmbeddings
#   * 阿里云通义千问嵌入、腾讯云向量嵌入	 

# In[31]:


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


# ### BGE Model
# https://huggingface.co/BAAI/bge-small-zh

# 使用embedding模型持久化存储，目前常用的中文模型是bge-large-zh-v1.5

# In[32]:


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
    


# In[33]:


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


# ## Vector_store
# ### Chroma
# 
# 优势是以最小成本实现向量数据库的核心价值—— 无需关注底层细节，快速搭建可用的向量检索系统，尤其适合原型开发、中小规模应用或对部署复杂度敏感的场景。

# In[34]:


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings_chroma(embedding_name, chunks, persist_dir=os.path.join(project_dir,"db/chroma_db")):
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


# In[35]:


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


# In[36]:


def get_serarch(vecotr_db):
    results = vecotr_db.similarity_search_with_score(
        "杜甫", k=3, 
        )
    # print(results)
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


# In[37]:


@skip_execution(IS_SKIP)
def test_chroma():
    path = os.path.join(project_dir,"datasets/tangshi.pdf") 
    vector_path =os.path.join(project_dir,"db/chroma_db") 

    data = load_document(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    print(len(chunks))
    create_embeddings_chroma(embedding_name,chunks,vector_path)
    vecotr_db = load_embeddings_chroma(embedding_name,vector_path)
    get_serarch(vecotr_db)
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

# In[38]:


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


# Faiss 分批次保存 避免高内存占用

# In[39]:


def create_embeddings_faiss(
    embedding_name,
    chunks,
    persist_dir=os.path.join(project_dir, "db/faiss_db"),
    batch_size=1000
):
    """
    使用FAISS向量数据库，分批次保存
    """
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


# In[40]:


@skip_execution(IS_SKIP)
def test_faiss():
    path = os.path.join(project_dir,"datasets/tangshi.pdf") 
    vector_path =os.path.join(project_dir,"db/faiss_db") 
    data = load_document(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    create_embeddings_faiss(embedding_name,chunks,vector_path,100)
    vecotr_db = load_embeddings_faiss(embedding_name,vector_path)
    get_serarch(vecotr_db)

test_faiss()


# In[41]:


@skip_execution(IS_SKIP)
def test_law():
    path = os.path.join(project_dir,"datasets","chinese_law_ft_dataset.json") 
    vector_path =os.path.join(project_dir,"db/law_db") 
    data = load_json(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    create_embeddings_faiss(embedding_name,chunks,vector_path,batch_size=500)
    load_embeddings_faiss(embedding_name,vector_path)
    
test_law()


# ### Milvus
# 
# * 高性能向量检索
# 
# 支持近似最近邻搜索（ANN），可处理亿级甚至百亿级向量
# 
# 多种索引算法（如 IVF、HNSW、DiskANN）可按需选择
# 
# * 分布式架构
# 
# 存储与计算分离，支持横向扩展
# 
# 云原生设计，适配 Kubernetes，易于部署和弹性伸缩
# 
# * 多模态数据支持
# 
# 支持稠密向量、稀疏向量、二进制向量
# 
# 可结合标量字段进行混合查询（如向量 + 标签过滤

# 
# ####  milvus服务端
# mkdir -p ~/milvus && cd ~/milvus
# wget https://github.com/milvus-io/milvus/releases/download/v2.5.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
# 
# #### 启动服务
# cd ~/milvus
# 
# sudo docker compose up -d
# 
# lsof -i:9000
# 
# sudo kill -9 <PID>
# #### 验证服务
# sudo docker compose ps
# 
# 
# ```
# milvus/volumes 文件夹
# etcd	元数据存储	保存 collection 定义、字段 schema、分区信息等。是 Milvus 的“脑袋”，负责协调和管理元信息。
# 
# minio	向量数据与索引存储	模拟对象存储（兼容 S3），保存 insert_log、index_log、delta_log 等。是 Milvus 的“硬盘”。
# 
# standalone	主服务节点	包含所有核心组件（QueryNode、DataNode、IndexNode、RootCoord 等），负责处理客户端请求、向量插入、检索、索引构建等。是 Milvus 的“大脑 + 手脚”。
# ```

# In[42]:


from pymilvus import Collection, MilvusException, connections, db, utility

@skip_execution(IS_SKIP)
def test_milvus_connect():
    try:
        connections.connect(host="localhost", port="19530")
        collections = utility.list_collections()

        print("✅ 成功连接 Milvus")
        print(collections)
    except Exception as e:
        print("❌ 连接失败：", e)

test_milvus_connect()


# In[43]:


def delete_collection(collection_name):
    """删除指定的 Milvus 集合"""
    
    if utility.has_collection(collection_name):
        try:
            utility.drop_collection(collection_name)
            print(f"集合 '{collection_name}' 已成功删除")
            
            # 验证删除结果
            if not utility.has_collection(collection_name):
                print(f"验证: 集合 '{collection_name}' 已不存在")
                return True
            else:
                print(f"警告: 集合 '{collection_name}' 似乎未被删除")
                return False
        except Exception as e:
            print(f"删除集合时发生错误: {e}")
            return False
    else:
        print(f"集合 '{collection_name}' 不存在，无需删除")
        return True
    
def create_milvus_database(db_name):
    conn = connections.connect(host="localhost", port="19530")

    # Check if the database exists
    db_name = db_name
    try:
        existing_databases = db.list_database()
        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")

            # Use the database context
            db.using_database(db_name)

            # Drop all collections in the database
            collections = utility.list_collections()
            for collection_name in collections:
                delete_collection(collection_name)

            db.drop_database(db_name)
            print(f"Database '{db_name}' has been deleted.")
        else:
            print(f"Database '{db_name}' does not exist.")
            
        database = db.create_database(db_name)
        print(f"Database '{db_name}' created successfully.")
    except MilvusException as e:
        print(f"An error occurred: {e}")
    return database


# In[44]:


from uuid import uuid4

def create_embeddings_milvus(
    embedding_name,
    chunks,
    db_name="milvus_db",
    collection_name="milvus_collection",
    batch_size=1000,
    milvus_host="localhost",
    milvus_port="19530"
):
    # 初始化 embedding 模型
    embeddings = get_embedding(embedding_name)
    connection_args={
                            "host": milvus_host,  # Milvus 服务地址
                            "port": milvus_port,      # Milvus 服务端口
                             "token": "root:Milvus",
                             "db_name": db_name
    }
    # 连接 Milvus 服务
    connections.connect(host=milvus_host, port=milvus_port)
    db.using_database(db_name)
    exists=  utility.has_collection(collection_name)
    print("Collection exists:", exists)
  
    if exists:
        print(f"检测到已有 Milvus collection：{collection_name}，尝试加载...")
        vector_store = Milvus(collection_name=collection_name, 
                              embedding_function=embeddings,
                              connection_args=connection_args,
                              auto_id=True)

        collection = Collection(collection_name)
        existing_count =collection.num_entities
        print(f"已加载 {existing_count} 条文档")
    else:
        print("创建新的 Milvus collection...")
        initial_batch = chunks[:batch_size]
        if not initial_batch:
            raise ValueError("chunks 为空，无法初始化 Milvus 向量库")
        vector_store = Milvus.from_documents(
            documents=initial_batch,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            )
        existing_count = len(initial_batch)

    # 边界检查
    if existing_count >= len(chunks):
        print("所有 chunks 已处理，无需更新 Milvus 向量库")
        return vector_store

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
        uuids = [str(uuid4()) for _ in range(len(batch))]
        vector_store.add_documents(documents=batch, ids=uuids)
     
    collection = Collection(collection_name)
    collection.flush()
    print(f"Milvus 向量库构建完成，共 {collection.num_entities} 条文档")
    return vector_store


# In[45]:


def load_embeddings_milvus(embedding_name,    
                            db_name="milvus_db",
                            collection_name="milvus_collection",
                            milvus_host="localhost",
                            milvus_port="19530"):
    """
    加载已保存的 Milvus 向量库
    """
    embeddings = get_embedding(embedding_name)
    connection_args={
                            "host": milvus_host,  # Milvus 服务地址
                            "port": milvus_port,      # Milvus 服务端口
                             "token": "root:Milvus",
                             "db_name": db_name
    }
    connections.connect(host=milvus_host, port=milvus_port)
    db.using_database(db_name)
    exists=  utility.has_collection(collection_name)
  
    if exists:
        print(f"检测到已有collection：{collection_name}，尝试加载...")
        vector_store = Milvus(collection_name=collection_name, 
                              embedding_function=embeddings,
                              connection_args=connection_args,
                              auto_id=True)
    
    print(f"Milvus 向量库已从 {db_name}的{collection_name} 加载")
    return vector_store


# In[46]:


@skip_execution(IS_SKIP)
def test_milvus():
    collection_name= 'milvus_collection'
    # create_milvus_database("milvus_db")
    path = os.path.join(project_dir,"datasets/tangshi.pdf") 
    data = load_document(path)
    chunks = chunk_data(data,chunk_size=512,chunk_overlap=100) 
    create_embeddings_milvus(embedding_name,chunks,"milvus_db",collection_name,10)
    vecotr_db = load_embeddings_milvus(embedding_name)
    get_serarch(vecotr_db)
    
test_milvus()

