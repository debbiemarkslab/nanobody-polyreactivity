from python:3
RUN python -m pip install --upgrade pip

WORKDIR /nanobody-polyreactivity

# Install python dependencies
COPY ./requirements.txt $WORKDIR
RUN python -m pip install -r ./requirements.txt
