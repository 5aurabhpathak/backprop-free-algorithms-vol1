FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt update && apt install -y tmux vim git libtcmalloc-minimal4 openssh-server
RUN python -m pip install -U pip && python -m pip install pipenv

RUN mkdir /blue
WORKDIR /blue

COPY Pipfile ./
RUN pipenv install --skip-lock --system --dev --verbose
RUN rm Pipfile Pipfile.lock
