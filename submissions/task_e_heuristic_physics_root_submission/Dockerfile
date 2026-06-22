FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/pytorch:2.7.1.8-cuda12.8.1-py312-alinux3.2104

ENV TZ=Asia/Shanghai
ENV PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ENV PIP_TRUSTED_HOST=mirrors.tuna.tsinghua.edu.cn

RUN set -x && \
    groupadd -g 1000 admin && \
    useradd -u 1000 -g admin -m admin && \
    mkdir -p /home/admin/appspace/atec2026/robot/solution/ && \
    chown -R admin:admin /home/admin/appspace && \
    chown -R admin:admin /opt/ac2

USER admin
WORKDIR /home/admin/appspace/atec2026/robot

COPY --chown=admin:admin solution.py ./solution/solution.py
COPY --chown=admin:admin server.py ./solution/server.py
COPY --chown=admin:admin run.sh ./solution/run.sh
COPY --chown=admin:admin requirements.txt ./solution/requirements.txt

RUN set -x && \
    chmod +x ./solution/run.sh && \
    ln -s /opt/ac2 venv

RUN set -x && \
    source ./venv/bin/activate && \
    pip install --no-cache-dir fastapi==0.136.0 uvicorn==0.45.0 python-multipart==0.0.26 && \
    if [ -s "solution/requirements.txt" ]; then \
        pip install --no-cache-dir -r solution/requirements.txt; \
    fi

CMD ["./solution/run.sh"]
