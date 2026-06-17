#!/bin/bash

inits(){
sudo apt install python3-venv -y
sudo apt isntall nvidia-smi

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

}

#for python 3.14.4 which was originally used
python_314(){
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
  libnss3-dev libssl-dev libreadline-dev libffi-dev wget

wget https://www.python.org/ftp/python/3.14.0/Python-3.14.0a7.tar.xz
tar -xf Python-3.14.0a7.tar.xz
cd Python-3.14.0a7
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
}

inits 