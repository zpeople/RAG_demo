# %%
import os
import sys
import time
from typing import Dict, Any, Mapping, Optional, List

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Field
from vllm import LLM as VLLM
from vllm import SamplingParams

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tool import skip_execution

# %%
MODEL_NAME ="Qwen/Qwen3-0.6B"
IS_SKIP=False

# %%

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
        model: str = None,
        temperature: float = 0.0,
        max_token: int = 2048,
        n_ctx: int = 512):
    """
    根据模型名称去加载模型，返回response数据
    """
    model_path = os.path.join("../model",model)
    print(model_path)
    if not os.path.exists(model_path):
        download_model(model_path,model)

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
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=n_ctx,
        max_model_len=10000,
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



# %%
@skip_execution(IS_SKIP)
def test_get_llm():
    outputs =get_llm_model("你是谁",MODEL_NAME,0.8,1024,512)
    print_outputs(outputs)

test_get_llm()

# %%
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
        model_path = os.path.join("../model",self.model_name)
        print("qwen_path:", model_path)
        
        self._llm = VLLM(
            model=model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=self.n_ctx,
            max_model_len=10000,
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


# %%


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


test_with_langchain()


