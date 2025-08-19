sudo apt-get -y update && sudo apt-get -y install tmux vim htop

tmux new -s base

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/aiscuser/miniconda3
eval "$(/home/aiscuser/miniconda3/bin/conda shell.bash hook)"
cd /scratch/amlt_code

wget https://aka.ms/downloadazcopy-v10-linux -O azcopy_linux_amd64.tar.gz && \
    tar xvf azcopy_linux_amd64.tar.gz && \
    sudo mv azcopy_linux_amd64*/azcopy /usr/local/bin && \
    chmod +x /usr/local/bin/azcopy && \
    rm -rf azcopy_linux_amd64*
# sudo apt-get install htop

# conda init
# source /home/aiscuser/.bashrc

conda tos accept
conda create --name molmo python=3.11 --y
conda activate molmo
pip install -e .[all]

pip install git+https://github.com/microsoft/azfuse.git


