# 构建 image
# docker build -t ragapp:1.0 .

# 构建容器
# docker run --gpus all --runtime=nvidia --name myapp -p 8501:8501 ragapp:1.0 streamlit run ./src/st_app.py



# 开启缓存共享s
# sudo apt remove docker-buildx-plugin
# mkdir -p ~/.docker/cli-plugins
# curl -L "https://github.com/docker/buildx/releases/download/v0.28.0/buildx-v0.28.0.linux-amd64" -o ~/.docker/cli-plugins/docker-buildx
# chmod +x ~/.docker/cli-plugins/docker-buildx
# export DOCKER_BUILDKIT=1

# 启动容器开始运行
# docker start myapp
# docker exec myapp pip install sentence-transformers #差pkg就单独装一下
# docker exec myapp streamlit run /app/src/st_app.py

# 更新文件到image
# docker cp ./src/Embedding.py  myapp:/app/src/Embedding.py
# docker commit myapp ragapp:1.0

FROM python:3.10

# 设置工作目录：后续命令将在这个目录下执行
WORKDIR /app

RUN pip install --upgrade pip

# 单独测试streamlit
# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
#         && pip install streamlit

# 复制文件：将当前目录下的 requirements.txt 复制到容器的 /app 目录
COPY requirements.txt .
# 安装依赖：运行 pip 命令安装所需的 Python 包  WSL中缓存共享
RUN  --mount=type=cache,target=/root/.cache/pip \
 pip install --no-cache-dir -r requirements.txt

# 复制应用代码：将当前目录下的所有文件复制到容器的 /app 目录
COPY . /app

# 加一个默认streamlit hello创建一个演示页面，映射端口以便网页访问  测试是否启动成功
CMD ["streamlit", "hello", "--server.port=8501", "--server.address=0.0.0.0"]