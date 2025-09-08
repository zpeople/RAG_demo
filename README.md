# RAG demo
基于Langchain测试 RAG 流程  


## 文件夹结构

    ├── RAG_demo
    │   ├── datasets                        # pdf 测试数据集
    │   ├── db
    │   │   ├── chroma_db                   # 向量数据库 chroma
    │   │   ├── faiss_db                    # 向量数据库 faiss
    │   │   └── law_db                      # 法律向量数据库 faiss 

    │   ├── model                           # 存储下载的模型
    │   ├── src
    │   │   ├── config    
    │   │   │   ├── install_pkgs.ipynb      # 配置环境
    │   │   │   └── ipynb2py.py             # 一键批量ipynb导出py                 
    │   │   ├── test                        # 测试代码
    │   │   ├── Embedding.ipynb             # 加载emb模型将chunk数据存为向量数据库  
    │   │   ├── Embedding.py                #（导出的py，readonly）
    │   │   ├── LoadData.ipynb              # 加载数据 chunk
    │   │   ├── LoadData.py                 #（导出的py，readonly）
    │   │   ├── models.ipynb                # langchain+vllm 构建模型生成逻辑
    │   │   ├── models.py                   #（导出的py，readonly）
    │   │   ├── st_app.py                   # streamlit 交互式 Web 测试模型 
    │   │   ├── tool.py                     # Python 装饰器
    │   │ 
    │   ├── .gitignore
    │   ├── README.md
    │   └── requirements.txt

## 核心逻辑

### 无论底层用的是 FAISS、Chroma 还是其他，LangChain 都会封装它们的相似性搜索逻辑，对外提供一致的检索接口

    # vector_db 是初始化的向量数据库实例（ Chroma、FAISS 或其他 LangChain 支持的向量存储）。
    # as_retriever() 是 LangChain 提供的统一接口，用于将向量数据库转换为 “检索器”（Retriever）
    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': top_k})

### LangChain 的 QA 链，作用是将 “检索器（Retriever）” 和 “大语言模型（LLM）” 串联起来

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt_template},
                                        return_source_documents=True)

## Embedding.ipynb  
* embedding model：BAAI/bge-small-zh
* vector_db:
保存和加载 Chroma 、Faiss、 Milvus 三种向量库
在modles里面测试三种不同向量库的搜索和召回


## Models.ipynb
本地部署:qwen 0.6b
在线模型:qwen plus