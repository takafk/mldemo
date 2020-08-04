FROM jupyter/datascience-notebook:7a0c7325e470

# Install additional packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/conda install -c anaconda py-xgboost -y --quiet \
    && conda install -c conda-forge nodejs

## JupyterLab

RUN jupyter labextension install @lckr/jupyterlab_variableinspector

RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter@1.2.0 \
    && jupyter serverextension enable --py jupyterlab_code_formatter

