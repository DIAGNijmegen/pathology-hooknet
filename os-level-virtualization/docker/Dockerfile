From tensorflow/tensorflow:latest-gpu


# Install ASAP-1.9 package from GitHub.
#

ARG asap_version=1.9
ARG asap_deb=ASAP-${asap_version}-Linux-Ubuntu1804.deb
RUN curl --remote-name --location https://github.com/computationalpathologygroup/ASAP/releases/download/${asap_version}/${asap_deb}
RUN apt-get update
RUN dpkg --install ${asap_deb} || true
RUN apt-get install --fix-broken --assume-yes
RUN ldconfig -v 
RUN echo "/opt/ASAP/bin" > /usr/local/lib/python3.6/dist-packages/asap.pth 
RUN rm ${asap_deb}


# Install HookNet from Github
#

ARG hooknet=https://github.com/computationalpathologygroup/hooknet/archive/master.zip
ADD ${hooknet} hooknet.zip
RUN unzip hooknet.zip -d /usr/local/lib/python3.6/dist-packages/
RUN mv /usr/local/lib/python3.6/dist-packages/hooknet-master /usr/local/lib/python3.6/dist-packages/hooknet
RUN rm hooknet.zip


# Install  requirements
COPY ./requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

WORKDIR /usr/local/lib/python3.6/dist-packages/hooknet