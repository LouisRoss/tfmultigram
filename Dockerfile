FROM    tensorflow/tensorflow:latest-gpu
LABEL   maintainer="Louis Ross <louis.ross@gmail.com"

ARG     MYDIR=/tensorflow
WORKDIR ${MYDIR}

COPY    install-deps ${MYDIR}/

RUN     echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN     bash ${MYDIR}/install-deps ${MYDIR} >>install-deps.log

RUN     pip install -r ${MYDIR}/requirements.txt

CMD     ["bash"]
