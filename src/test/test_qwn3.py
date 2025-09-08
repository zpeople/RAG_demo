
# echo "export DASHSCOPE_API_KEY='sk-'" >> ~/.bashrc
# source ~/.bashrc
# echo $DASHSCOPE_API_KEY

# 此处以qwen-plus为例，可按需更换模型名称。
# 调用文档：https://help.aliyun.com/zh/model-studio/chat/
# 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
import os
import asyncio
from openai import OpenAI
from openai import AsyncOpenAI

BASE_URL= "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME ="qwen-plus"
api_key =""
api_key = (
            api_key
            or os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
if not api_key:
    raise ValueError(
        "未找到 DashScope API‑KEY！请在实例化时传入 api_key "
        "或设置环境变量 QWEN_API_KEY / DASHSCOPE_API_KEY")

# 单轮对话 异步调用 流式输出
def one_chat(prompt,api_key):
    client = AsyncOpenAI(
        api_key= api_key,
        base_url= BASE_URL,
    )

    async def main():
        response =  client.chat.completions.create(
            messages=[{"role": "user", "content": f"{prompt}"}],
            model=MODEL_NAME,  
            stream=True,
            stream_options={"include_usage": True}
        )
        for chunk in response:
            # print(chunk)
            if chunk.choices:
                print(chunk.choices[0].delta.content)
        # print(response.choices[0].message.content)

    asyncio.run(main())




# 初始化对话历史
def multi_chat(prompt,api_key):
    client = AsyncOpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    )
    chat_history = [
        {"role": "system", "content": f"{prompt}"}
    ]

    async def chat():
        while True:
            user_input = input("你：")
            if user_input.lower() in ["exit", "quit"]:
                print("再见啦！")
                break

            # 添加用户输入到历史
            chat_history.append({"role": "user", "content": user_input})

            try:
                response = await client.chat.completions.create(
                    messages = chat_history,
                    model = MODEL_NAME,
                )
                reply = response.choices[0].message.content
                print(f"我是AI：{reply}")

                # 添加助手回复到历史
                chat_history.append({"role": "assistant", "content": reply})

            except Exception as e:
                print(f"⚠️ 出错了：{e}")


    asyncio.run(chat())


one_chat('你是一个AI小助手',api_key)