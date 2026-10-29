
FROM apache/spark:3.5.1

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    pandas \
    numpy \
    pymongo \
    python-dotenv \
    pyspark==3.5.1 \
    streamlit==1.38.0 \
    plotly

USER spark
