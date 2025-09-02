# %% [markdown]
# 加载文件：使用langchain下的document_loaders加载pdf、docs、txt、md等格式文件
# 
# 文本分块：分块的方式有很多，选择不同的分块方法、分块大小、chunk_overlap，对最后的检索结果有影响

# %%
import os
from tool import skip_execution
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

IS_SKIP =False

# %%



def load_document(file):
    """
    加载PDF、DOC、TXT文档
    :param file:
    :return:
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
    else:
        print('Document format is not supported!')
        return None
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
    chunks = text_splitter.split_documents(data)
    return chunks


# %%
@skip_execution(IS_SKIP)
def test_load():
    path = "../datasets/test.pdf"
    data = load_document(path)
    print("data pages:",len(data))
    return data

data = test_load()
data

# %%
@skip_execution(IS_SKIP)
def test_chunk(data):
    chunks = chunk_data(data) 
    print("chunks len:",len(chunks))
    return chunks
    
test_chunk(data)


