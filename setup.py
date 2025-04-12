from setuptools import setup, find_packages

setup(
    name="novax",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.2.2",
        "transformers==4.48.0",
        "peft==0.7.1",
        "datasets==2.21.0",
        "tokenizers>=0.15.0",  # 添加 tokenizers 依赖
    ],
    author="你的名字",
    author_email="你的邮箱",
    description="NovaX: A lightweight chatbot library based on MiniMind",
    url="https://github.com/你的用户名/NovaX",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
