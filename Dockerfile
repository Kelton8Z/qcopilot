FROM python:3.11
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["grun", "bash", "p.sh", "streamlit","run", "main.py", "--server.port","8501","--server.fileWatcherType","none"]
