import re
import os
import sys
try:
    get_ipython
    current_dir = os.getcwd()
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Set path，temporary path expansion
project_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 原始文件路径
input_file = os.path.join( project_dir,"requirements.txt")
# 清理后的输出文件路径
output_file = os.path.join( project_dir,"requirements_c.txt")

# 可选：你想保留的核心依赖（留空则保留全部非开发包）
core_packages = {
    "fastapi", "torch", "transformers", "pandas", "scikit-learn", "streamlit",
    "openai", "matplotlib", "numpy", "uvicorn",
    "langchain", "langchain-community", "langchain-chroma", "langchain-huggingface",
    "torch", "torchvision", "torchaudio", "vllm","transformer","pypdf","sentence_transformers"

}

# 常见的开发/测试/Notebook工具包
dev_keywords = [
    "pytest", "ipython", "jupyter", "nbconvert", "nbformat", "debugpy",
    "coverage", "black", "mypy", "pylint", "flake8", "isort", "jupyterlab"
]

def is_local_or_editable(line):
    return line.startswith("-e ") or "file://" in line or re.match(r"\s*\.\s*", line)

def is_dev_package(line):
    return any(dev in line.lower() for dev in dev_keywords)

def is_core_package(line):
    pkg = line.split("==")[0].strip().lower()
    return pkg in core_packages

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        line = line.strip()
        if not line or is_local_or_editable(line) or is_dev_package(line):
            continue
        if core_packages and not is_core_package(line):
            continue
        outfile.write(line + "\n")

print(f"✅ 清理完成！输出文件已保存为：{output_file}")
