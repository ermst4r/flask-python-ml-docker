FROM tiangolo/uwsgi-nginx-flask:python3.5
RUN apt-get update && \
    apt-get install -y \
					git \
					curl \
					wget \
					unzip