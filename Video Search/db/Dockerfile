FROM postgres:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-all \
    && rm -rf /var/lib/apt/lists/*

COPY pgvector /pgvector

RUN cd /pgvector && \
    make && \
    make install

# FROM postgres:latest

# RUN apt-get update && \
#     apt-get install -y build-essential postgresql-server-dev-all

# RUN wget https://github.com/ankane/pgvector/archive/refs/tags/v0.1.0.tar.gz && \
#     tar -xzvf v0.1.0.tar.gz

# RUN cd pgvector-0.1.0 && \
#     make && \
#     make install

# # 在 PostgreSQL 数据库中加载 pgvector 扩展
# RUN echo "CREATE EXTENSION pgvector;" | su - postgres -c psql