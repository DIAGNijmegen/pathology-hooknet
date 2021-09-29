From tensorflow/tensorflow:latest-gpu


# Install python
#
ARG python_version=python3.8
RUN apt-get update
RUN apt-get install software-properties-common --assume-yes
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install ${python_version} --assume-yes
RUN apt-get install libpython3.8-dev --assume-yes


# Install Node.js (for notebook iwidgets)
#
RUN apt-get -y install curl dirmngr apt-transport-https lsb-release ca-certificates --assume-yes
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get -y install nodejs --assume-yes


# Install git
#
RUN  apt-get install git --assume-yes


# Install ASAP-1.9 package from GitHub.
#
ARG asap_version=1.9
ARG asap_deb=ASAP-${asap_version}-Linux-Ubuntu1804.deb
RUN curl --remote-name --location https://github.com/computationalpathologygroup/ASAP/releases/download/${asap_version}/${asap_deb}
RUN apt-get update
RUN dpkg --install ${asap_deb} || true
RUN apt-get install --fix-broken --assume-yes
RUN ldconfig -v 
RUN echo "/opt/ASAP/bin" > /usr/local/lib/${python_version}/dist-packages/asap.pth 
RUN rm ${asap_deb}


# Install openslide-tools
RUN  apt-get install openslide-tools


# Install Packages
#
RUN ${python_version} -m pip install six 
RUN ${python_version} -m pip install --upgrade pip
RUN ${python_version} -m pip install hooknet wholeslidedata jupyterlab ipywidgets
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN ${python_version} -m pip install --upgrade tensorflow-gpu==2.3  


# Set WORKDIR
#
WORKDIR /home/user/


# git clone HookNet
#
RUN cd /home/user/
RUN git clone https://github.com/DIAGNijmegen/pathology-hooknet