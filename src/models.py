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

import time
from typing import Dict, Any, Mapping, Optional, List

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Field
from vllm import LLM as VLLM
from vllm import SamplingParams

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from tool import skip_execution
from Embedding import load_embeddings_faiss,load_embeddings_chroma,load_embeddings_milvus


# In[ ]:


MODEL_NAME ="Qwen/Qwen3-0.6B"
IS_SKIP=True


# In[ ]:


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"输入: {prompt}")
        print(f"输出: {generated_text}\n")
    print("-" * 80)


def download_model(localpath,modelname):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=modelname, 
        local_dir=localpath,
        local_dir_use_symlinks=False,  # Windows 必加
        allow_patterns=["*.json", "*.bin", "*.txt", "*.model"] 
    )
    
def get_llm_model(
        prompt: str = None,
        model_name: str = None,
        temperature: float = 0.0,
        max_token: int = 2048,
        n_ctx: int = 512):
    """
    根据模型名称去加载模型，返回response数据
    """
    model_path = os.path.join(project_dir,"model",model_name)
    print(model_path)
    if not os.path.exists(model_path):
        download_model(model_path,model_name)

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_token,
        top_p=0.95
    )
    
    # 初始化VLLM
    llm = VLLM(
        model=model_path,
        tensor_parallel_size=1,  # 根据GPU数量调整
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=n_ctx,
        max_model_len=n_ctx,
    )
    
    start = time.time()
    # 生成响应
    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "你是一个智能超级助手，请用专业的词语回答问题，整体上下文带有逻辑性，如果不知道，请不要乱说",
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        sampling_params=sampling_params
    )
    
    cost = time.time() - start
    print(f"模型生成时间：{cost}")
    print(f"大模型回复：\n{response}")
    return response



# In[ ]:


@skip_execution(IS_SKIP)
def test_get_llm():
    outputs =get_llm_model("你是谁",MODEL_NAME,0.8,1024,512)
    print_outputs(outputs)

# test_get_llm()


# In[ ]:


class QwenLLM(LLM):
    """
    基于VLLM的自定义QwenLLM
    """
    model_name: str = ""
    # 访问时延上限
    request_timeout: float = None
    # 温度系数
    temperature: float = 0.8
    # 窗口大小
    n_ctx :int =2048
    # token大小
    max_tokens:int= 1024
    # 并行计算数量
    tensor_parallel_size: int = 1
    # 模型参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # VLLM实例
    _llm: Optional[VLLM] = None

    def __init__(self, **data: Any):
        super().__init__(** data)
        self._initialize_llm()

    def _initialize_llm(self):
        """初始化VLLM实例"""
        model_path = os.path.join(project_dir,"model",self.model_name)
        print("qwen_path:", model_path)
        
        self._llm = VLLM(
            model=model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.8,
            max_num_batched_tokens=self.n_ctx,
            max_model_len=self.n_ctx,
            **self.model_kwargs
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              ** kwargs: Any):
        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
            stop=stop or []
        )
        
        # 生成响应
        response = self._llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个智能超级助手，请用[中文]专业的词语回答问题，整体上下文带有逻辑性，并以markdown格式输出",
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            sampling_params=sampling_params
        )

        print(f"Qwen response: \n{response}")
        return response[0].outputs[0].text

    @property
    def _llm_type(self) -> str:
        return "vllm-qwen"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            "n_ctx": self.n_ctx,
            "max_tokens": self.max_tokens,
            "tensor_parallel_size": self.tensor_parallel_size
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}


# In[ ]:


@skip_execution(IS_SKIP)
def test_with_langchain():
    # 初始化模型
    llm = QwenLLM(
        model_name=MODEL_NAME,
        temperature=0.5
    )

    # 创建一个简单的链
    prompt = PromptTemplate(
        input_variables=["question"],
        template="请回答以下问题: {question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # 运行测试
    result = chain.run("什么是机器学习？")
    print(f"LangChain测试结果: {result}")


# test_with_langchain()


# In[ ]:


# 从标志结束的位置开始截取
def extract_after_flag(text, flag):
    # 找到标志的起始索引
    index = text.find(flag)
    if index != -1:
        return text[index + len(flag):]
    return ""  # 如果没有找到标志，返回空字符串



def ask_and_get_answer_from_local(model_name, vector_db, prompt,template, top_k=5):
    """
    从本地加载大模型
    """
    llm = QwenLLM(model_name=model_name, temperature=0.4)
    # region 测试用
    if not IS_SKIP:#  创建基础提示模板（无上下文） 直接生成回答 测试的时候用来对比输出
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=template.replace("{context}\n", "")  # 移除上下文占位符
        )
        
        prompt_text = prompt_template.format(question=prompt)
        direct_answer = llm(prompt_text)  
        print(f"direct answers: {direct_answer}")
    # endregion

    # RAG 
    docs_and_scores = vector_db.similarity_search_with_score(prompt, k=top_k)
    print("docs_and_scores: ", docs_and_scores)
    # knowledge = [doc["page_content"] for doc in docs_and_scores] # 根据不同模型要调整字段
    # print("检索到的知识：", knowledge)

    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': top_k})
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt_template},
                                        return_source_documents=True)
    answer = chain({"query": prompt, "top_k": top_k})
    # answer = chain.run(prompt)
    # answer = answer['choices'][0]['message']['content']
    answer = answer['result']
    print(f"answers: {answer}")
    answer =extract_after_flag(answer,"</think>")
    return answer


# In[ ]:


DEFAULT_TEMPLATE = """
        你是一个聪明的超级智能助手，请用专业且富有逻辑顺序的句子回复，并以中文形式且markdown形式输出。
        检索到的信息：
        {context}
        问题：
        {question}
    """
prompt ="'少小离家老大回'的下一句诗是什么"

embedding_name ="BAAI/bge-small-zh"

@skip_execution(IS_SKIP)
def test_getRagAnswer_faiss():
    faiss_db = load_embeddings_faiss(embedding_name,vector_db_path=os.path.join(project_dir,"db/faiss_db"))
    ask_and_get_answer_from_local(MODEL_NAME,faiss_db,prompt,DEFAULT_TEMPLATE)

@skip_execution(IS_SKIP)
def test_getRagAnswer_chroma():
    chroma_db = load_embeddings_chroma(embedding_name,persist_dir=os.path.join(project_dir,"db/chroma_db"))
    ask_and_get_answer_from_local(MODEL_NAME,chroma_db,prompt,DEFAULT_TEMPLATE)


@skip_execution(IS_SKIP)
def test_getRagAnswer_milvus():
    milvus_db = load_embeddings_milvus(embedding_name)
    ask_and_get_answer_from_local(MODEL_NAME,milvus_db,prompt,DEFAULT_TEMPLATE)

test_getRagAnswer_chroma()


# In[ ]:


from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
from pydantic import Field
BASE_URL= "https://dashscope.aliyuncs.com/compatible-mode/v1"
ONLINE_MODEL_NAME ="qwen-plus"


class OpenAILLM(LLM):
    """
    一个将OpenAI API包装为LangChain LLM的自定义类
    """
    api_key: Optional[str] = Field(None)
    base_url:Optional[str] = Field(BASE_URL)
    model_name: str = Field(ONLINE_MODEL_NAME)
    temperature: float = Field(0.7)
    max_tokens: int = Field(1024)
    
    def __init__(self, **data: Any):
        super().__init__(** data)
        # 配置OpenAI API密钥
        if self.api_key:
            OpenAI.api_key = self.api_key
         
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return self.model_name

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """返回用于标识LLM的参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        实现调用OpenAI API的核心方法
        """
        try:
            # 调用OpenAI的 completions API
            client = OpenAI(
            api_key= self.api_key,
            base_url=self.base_url,
            )
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个智能超级助手，请用[中文]专业的词语回答问题，整体上下文带有逻辑性，并以markdown格式输出",
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
               temperature=self.temperature,
               max_tokens= self.max_tokens
            )
            # 提取并返回生成的文本
            return completion.choices[0].message.content
            
        except Exception as e:
            raise ValueError(f"调用OpenAI API时发生错误: {str(e)}")

    def predict(self, text: str, **kwargs: Any) -> str:
        """预测方法，与LangChain其他组件兼容"""
        return self._call(text,** kwargs)



# In[ ]:


def ask_and_get_answer(model_name,url,vector_db, prompt,template,top_k=3,api_key=None ):
    api_key = (
            api_key
            or os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )

    llm =  OpenAILLM(model_name=model_name,base_url=url,api_key=api_key)
    # region 测试用
    if not IS_SKIP:
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=template.replace("{context}\n", "")  # 移除上下文占位符
        )
        
        prompt_text = prompt_template.format(question=prompt)
        direct_answer = llm(prompt_text)  
        print(f"direct answers: {direct_answer}")
    # endregion
    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': top_k})
    prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt_template},
                                        return_source_documents=True)
    answer = chain({"query": prompt, "top_k": top_k})
    answer = answer['result']
    print(f"answers: {answer}")
    return answer


# In[ ]:


@skip_execution(IS_SKIP)
def test_getRagAnswer_chroma():
    chroma_db = load_embeddings_chroma(embedding_name,persist_dir=os.path.join(project_dir,"db/chroma_db"))
    ask_and_get_answer(ONLINE_MODEL_NAME,BASE_URL,chroma_db,prompt,DEFAULT_TEMPLATE)

test_getRagAnswer_chroma()


# In[ ]:


prompt ="杀人自首以后，怎么判刑"
@skip_execution(IS_SKIP)
def test_getRagAnswer_law():
    chroma_db = load_embeddings_faiss(embedding_name,persist_dir=os.path.join(project_dir,"db/law_db"))
    ask_and_get_answer_from_local(MODEL_NAME,chroma_db,prompt,DEFAULT_TEMPLATE)

test_getRagAnswer_law()

