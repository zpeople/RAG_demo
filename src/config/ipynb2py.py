import os
import subprocess
from pathlib import Path
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


def batch_convert_ipynb_to_py(input_dir, output_dir=None):
    """
    批量将指定目录下的.ipynb文件转换为.py文件
    
    参数:
        input_dir: 包含.ipynb文件的目录
        output_dir: 输出.py文件的目录，默认为与.ipynb文件相同的目录
    """
    # 确保输入目录存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在")
        return
    
    # 获取目录下所有.ipynb文件
    ipynb_files = list(Path(input_dir).glob("*.ipynb"))
    
    if not ipynb_files:
        print(f"在 '{input_dir}' 中未找到任何.ipynb文件")
        return
    
    # 确保输出目录存在
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 逐个转换文件
    for ipynb_path in ipynb_files:
        try:
            # 构建输出路径
            if output_dir:
                py_filename = os.path.basename(ipynb_path).replace(".ipynb", "")
                py_path = os.path.join(output_dir, py_filename)
                print(py_path)
                cmd = ["jupyter", "nbconvert", "--to", "script", str(ipynb_path), "--output", py_path]
            else:
                cmd = ["jupyter", "nbconvert", "--to", "script", str(ipynb_path)]
            
            # 执行转换命令
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"已转换: {ipynb_path}")
        except subprocess.CalledProcessError as e:
            print(f"转换失败 {ipynb_path}: {e.stderr}")
        except Exception as e:
            print(f"处理 {ipynb_path} 时出错: {str(e)}")
    
    print(f"转换完成，共处理 {len(ipynb_files)} 个文件")

if __name__ == "__main__":
    # 示例用法
    input_directory = os.path.join(project_dir,"src")  # 替换为你的.ipynb文件所在目录
    output_directory =os.path.join(project_dir,"src")  # 替换为你想要保存.py文件的目录，或设为None使用源目录
    print(output_directory)
    batch_convert_ipynb_to_py(input_directory, output_directory)
