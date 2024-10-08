conda create -n unsloth python=3.10 -y
conda activate unsloth
pip install unsloth pypdf langchain langchain_community langchain_openai streamlit streamlit_chat langchain_huggingface
conda install -c conda-forge faiss-cpu -y
conda create -n vllm python=3.10 -y 
conda activate vllm
pip install vllm pypdf langchain langchain_community langchain_openai streamlit streamlit_chat langchain_huggingface
conda install -c conda-forge faiss-cpu -y
