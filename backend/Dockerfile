from continuumio/anaconda3
RUN python -m pip install --upgrade pip

WORKDIR /nanobody-polyreactivity

# Install python dependencies
COPY ./requirements.txt $WORKDIR
RUN python -m pip install -r ./requirements.txt
RUN conda install -c bioconda anarci
RUN conda install -c bioconda hmmer
RUN conda install -c conda-forge biopython
# for macbook version
#RUN conda install pytorch torchvision torchaudio -c pytorch
# for linux aws version
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

