FROM pytorch/conda-builder:cpu AS base

COPY requirements.txt /
RUN /opt/conda/bin/pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/bin/pip install -r /requirements.txt

# ----
FROM base as models

COPY models.txt download_models.py /
RUN /opt/conda/bin/python /download_models.py

# ---
FROM final
WORKDIR /sandbox
COPY . /sandbox/

ENTRYPOINT [ "./entrypoint.sh" ]
