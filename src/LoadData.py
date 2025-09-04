#!/usr/bin/env python
# coding: utf-8

# 加载文件：使用langchain下的document_loaders加载pdf、docs、txt、md等格式文件
# 
# 文本分块：分块的方式有很多，选择不同的分块方法、分块大小、chunk_overlap，对最后的检索结果有影响

# In[118]:


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

from tool import skip_execution
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,RecursiveJsonSplitter
import json
IS_SKIP =True


# In[119]:


def load_document(file):
    """
    加载PDF、DOC、TXT文档
    """
    name, extension = os.path.splitext(file)
    if extension == '.pdf':
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        loader = UnstructuredFileLoader(file)
        print(f'Loading {file}')
    data = loader.load()
    return data

    
def chunk_data(data, chunk_size=256, chunk_overlap=150):
    """
    将数据分割成块
    :param data:
    :param chunk_size: chunk块大小
    :param chunk_overlap: 重叠部分大小
    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(len(data))
    chunks = text_splitter.split_documents(data)
    return chunks


# In[120]:


@skip_execution(IS_SKIP)
def test_load():
    path =os.path.join(project_dir,"datasets","test.pdf")

    print(os.path.abspath(path))
    data = load_document(path)
    print("data pages:",len(data))
    return data

data = test_load()
data


# In[121]:


@skip_execution(IS_SKIP)
def test_chunk(data):
    chunks = chunk_data(data) 
    print("chunks len:",len(chunks))
    return chunks
    
test_chunk(data)


# ### Load Json to document

# 整理JSON格式

# In[ ]:


import re
from typing import List, Dict, Any
def parse_legal_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析原始法律问答数据，转换为适合RAG系统使用的结构化格式
    
    参数:
        raw_data: 原始JSON数据
        
    返回:
        结构化后的法律问答数据
    """
    # 优先获取input字段
    user_query = raw_data.get('input', '').strip()
    
    # 如果input为空，则使用instruction字段
    if not user_query:
        user_query = raw_data.get('instruction', '').strip()
        
    # 移除多余的标点和重复字符
    user_query = re.sub(r'[\n\r]+', ' ', user_query)
    user_query = re.sub(r' +', ' ', user_query)
    user_query = re.sub(r'，+', '，', user_query)
    user_query = re.sub(r'？+', '？', user_query)
    
    # 处理回答内容，分离回答和法律依据
    output = raw_data.get('output', '').strip()
    
    # 提取回答部分
    answer_match = re.match(r'回答:(.*?)法律依据:', output, re.DOTALL)
    answer = ''
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # 如果没有明确分隔，尝试提取到第一个法律条文前
        # answer = re.split(r'《\w+法》', output, 1)[0].replace('回答:', '').strip()
        answer =output
    
    # 提取法律依据部分
    legal_basis_text = re.sub(r'^回答:.*?法律依据:', '', output, flags=re.DOTALL).strip()
    legal_basis = []
    
    # 正则匹配法律条文（如《民事诉讼法》第二百四十三条）
    law_pattern = re.compile(r'《(.*?)》(第?\s*[\d一二三四五六七八九十]+条?)\s*规定?，?(.*?)(?=《|$)', re.DOTALL)
    matches = law_pattern.findall(legal_basis_text)
    
    for match in matches:
        law_name, article, content = match
        # 清理内容
        content = content.strip().replace('\n', ' ')
        content = re.sub(r' +', ' ', content)
        
        legal_basis.append({
            "law_name": law_name.strip(),
            "article": article.strip().replace('第', '').replace('条', ''),
            "content": content
        })
    
    # 构建结构化数据
    structured_data = {
        "user_query": user_query,
        "answer": answer,
        "legal_basis": legal_basis
    }
    meta_data={ "id": raw_data.get('id', '')
    }
   
    return structured_data,meta_data

def batch_process(raw_data_list: List[Dict[str, Any]]) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """批量处理数据列表，分别返回结构化数据和元数据"""
    structured_list = []
    meta_list = []
    
    for data in raw_data_list:
        structured, meta = parse_legal_data(data)
        structured_list.append(structured)
        meta_list.append(meta)
    
    return structured_list, meta_list


# In[ ]:


from langchain.schema import Document


def load_json(file):
    """
    加载JSON
    """
    name, extension = os.path.splitext(file)
    if extension == '.json':
        # 处理JSON文件
        try:
            with open(file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                # 将JSON数据转换为Document 1json-->1 document
                # json_content = json.dumps(json_data, ensure_ascii=False, indent=2)
                # data = [Document(
                #     page_content=json_content,
                #     metadata={'source': file}
                # )]

                # 按每条instruction 拆分
                json_data,meta_data = batch_process(json_data)
                datas = [Document(page_content=str(chunk), metadata=meta) for chunk, meta in zip(json_data, meta_data)]
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    else:
        print('Document format is not supported!')
        return None

    print(f"pages: {len(datas)}")

    return datas


# In[ ]:


@skip_execution(IS_SKIP)
def test_load_json():
    path =os.path.join(project_dir,"datasets","chinese_law_ft_dataset_mini.json") 
    print(os.path.abspath(path))
    data = load_json(path)
    print("data pages:",len(data))
    return data


data=test_load_json()
print(type(data))


# In[125]:


test_chunk(data)

