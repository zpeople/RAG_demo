import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# 设置页面标题
st.title("我的第一个 Streamlit 应用")

# 添加文本
st.write("这是一个简单的数据分析工具")

# 上传文件组件
uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")

if uploaded_file is not None:
    # 读取数据
    df = pd.read_csv(uploaded_file)

    # 显示数据
    st.subheader("数据预览")
    st.dataframe(df)

    # 显示统计信息
    st.subheader("统计摘要")
    st.write(df.describe())

    # 绘制图表
    st.subheader("数据可视化")
    column = st.selectbox("选择列", df.columns)
    plt.hist(df[column])
    st.pyplot(plt)
