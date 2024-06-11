FROM python:3.10-slim

WORKDIR /Genie

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Nehaverma43/Genie.git .

RUN pip3 install -r requirements.txt

EXPOSE 8505



ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8505", "--server.address=0.0.0.0"]
