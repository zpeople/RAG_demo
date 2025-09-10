import os
import re
import sys

try:
    get_ipython
    current_dir = os.getcwd()
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置项目路径
project_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 原始文件路径
input_file = os.path.join(project_dir, "requirements.txt")
# 清理后的输出文件路径
output_file = os.path.join(project_dir, "requirements_c.txt")

# 核心依赖包（留空则保留全部非开发包）
core_packages = {
    "fastapi",
    "torch",
    "transformers",
    "pandas",
    "scikit-learn",
    "streamlit",
    "fire",
    "openai",
    "matplotlib",
    "numpy",
    "uvicorn",
    "langchain",
    "langchain-community",
    "langchain-chroma",
    "langchain-huggingface",
    "langchain-milvus",
    "pymilvus",
    "torch",
    "torchvision",
    "torchaudio",
    "vllm",
    "transformer",
    "pypdf",
    "sentence_transformers",
    "unsloth",
    "unsloth-zoo",
    "bitsandbytes",
    "accelerate",
    "xformers",
    "peft",
    "trl",
    "triton",
    "cut_cross_entropy",
}

# 常见的开发/测试/Notebook工具包关键词
dev_keywords = [
    "pytest",
    "ipython",
    "jupyter",
    "nbconvert",
    "nbformat",
    "debugpy",
    "coverage",
    "black",
    "mypy",
    "pylint",
    "flake8",
    "isort",
    "jupyterlab",
]


def is_local_or_editable(line):
    """判断是否为本地编辑模式安装的包（排除）"""
    return line.startswith("-e ") or re.match(r"\s*\.\s*", line)


def is_dev_package(line):
    """判断是否为开发依赖包（排除）"""
    return any(dev in line.lower() for dev in dev_keywords)


def is_core_package(line):
    """判断是否为需要保留的核心包"""
    # 提取包名（处理普通格式和Git链接格式）
    if "@" in line:
        # 处理Git链接格式，如 "unsloth @ git+https://..."
        pkg_name = line.split("@")[0].strip().lower()
    else:
        # 处理普通格式，如 "package==1.0.0"
        pkg_name = line.split("==")[0].strip().lower()
    return pkg_name in core_packages


def is_git_url_package(line):
    """判断是否为Git链接形式的包（保留）"""
    return "@ git+" in line or "git+https" in line or "git+ssh" in line


with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        line = line.strip()
        # 跳过空行、本地编辑模式包、开发包
        if not line or is_local_or_editable(line) or is_dev_package(line):
            continue

        # 保留核心包 或 Git链接形式的包
        if is_core_package(line) or is_git_url_package(line):
            outfile.write(line + "\n")

print(f"✅ 清理完成！输出文件已保存为：{output_file}")
