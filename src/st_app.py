import os
import sys
import time
try:
    get_ipython
    current_dir = os.getcwd()
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置路径，临时扩展路径
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import streamlit as st
import models as M
import Embedding as emb
from LoadData import load_document, chunk_data
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


emb_name = "BAAI/bge-small-zh"
model_name = "Qwen/Qwen3-0.6B"
online_model_name ="qwen-plus"
online_URL= "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEMPLATE = """
        你是一个聪明的超级智能助手，请用专业且富有逻辑顺序的句子回复，并以中文形式且markdown形式输出。
        检索到的信息：
        {context}
        问题：
        {question}
    """

# 运行应用: streamlit run ./st_app.py

vector_db_path = os.path.join(project_dir, "db", "law_db")
print(f"vector_db_path: {vector_db_path}")


# 清除streamlit会话状态中的聊天历史
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    if 'messages' in st.session_state:
        del st.session_state['messages']

# 流式输出生成器
def stream_answer_generator(model_name, vector_db, prompt, top_k):
    """生成流式响应的生成器函数"""
    # 获取相关文档
    docs = vector_db.similarity_search(prompt, k=top_k)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 构建完整提示
    full_prompt = f"基于以下上下文回答问题：\n{context}\n\n问题：{prompt}\n\n回答："
    
    # 逐步生成响应
    response = ""
    for chunk in M.generate_streaming_response(model_name, full_prompt):
        response += chunk
        yield chunk
        # 模拟流式输出的延迟（实际使用中可去掉）
        # time.sleep(0.05)


if __name__ == "__main__":
    # 页面设置
    st.set_page_config(page_title="LLM问答应用", page_icon="🤖")
    
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
       
        # 选择本地大模型
        llm = st.selectbox(
            label="请选择大模型",
            options=(model_name,online_model_name, )
        )
        
        # 选择向量数据库
        embedding = st.selectbox(
            "请选择向量数据库",
            ('FAISS', 'Chroma')
        )

        # 文件上传组件
        uploaded_file = st.file_uploader('上传文件:', type=['pdf', 'docx', 'txt'])

        #  chunk大小设置
        chunk_size = st.number_input(
            'chunk_size:', 
            min_value=100, 
            max_value=2048, 
            value=512, 
            on_change=clear_history
        )

        # chunk重叠设置
        chunk_overlap = st.number_input(
            label="chunk_overlap", 
            min_value=0, 
            max_value=1024, 
            value=150,
            on_change=clear_history
        )

        # 检索数量设置
        k = st.number_input(
            'top_k', 
            min_value=1, 
            max_value=20, 
            value=3, 
            on_change=clear_history
        )

        # 添加数据按钮
        add_data = st.button('添加数据', on_click=clear_history)

       

        if uploaded_file and add_data:  # 如果用户上传了文件
            with st.spinner('正在读取、分割和嵌入文件...'):
                # 将文件从RAM写入当前目录
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.write(f'Chunk大小: {chunk_size}, 重叠度: {chunk_overlap}, 总片段数: {len(chunks)}')

                # 创建嵌入并返回向量存储
                if embedding == "FAISS":
                    vector_store = emb.create_embeddings_faiss(
                        embedding_name=emb_name,
                        chunks=chunks,
                        # vector_db_path=vector_db_path, 
                    )
                elif embedding == "Chroma":
                    vector_store = emb.create_embeddings_chroma(emb_name,chunks)

                # 将向量存储保存在streamlit会话状态中
                st.session_state.vs = vector_store
                st.success('文件上传、分割和嵌入成功。')

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 展示对话历史
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg["content"])

    # 加载默认向量数据库（如果尚未加载）
    if 'vs' not in st.session_state:
        try:
            with st.spinner('正在加载向量数据库...'):
                vector_store = emb.load_embeddings_faiss(emb_name,vector_db_path)
                st.session_state.vs = vector_store
                st.toast('向量数据库加载成功!', icon='😍')
        except Exception as e:
            st.warning(f'加载向量数据库失败: {str(e)}', icon="⚠️")

    # 处理用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        # 将用户消息添加到对话历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 获取向量存储
        vector_store = st.session_state.get('vs')
        
        if vector_store is not None:
            # 本地模型
            if llm == model_name:
                with st.spinner('正在生成回答...'):
                    response = M.ask_and_get_answer_from_local(
                        model_name=model_name, 
                        vector_db=vector_store, 
                        prompt=prompt, 
                        template=DEFAULT_TEMPLATE,
                        top_k=k
                    )
                    
                    # 显示助手回答
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    # 添加到对话历史
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
            # 在线模型
            elif llm == online_model_name:
               with st.spinner('正在生成回答...'):
                    response = M.ask_and_get_answer(online_model_name, 
                                            url=online_URL,
                                            vector_db=vector_store, 
                                            prompt=prompt, 
                                            template=DEFAULT_TEMPLATE,
                                            top_k=k,
                                            )
                    # 显示助手回答
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    # 添加完整响应到对话历史
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning('请先添加数据或等待向量数据库加载完成', icon="⚠️")
    