#bin/bash
sudo mkfs.ext4 -E nodiscard /dev/nvme0n1
mkdir ~/storage
sudo mount /dev/nvme0n1 ~/storage
sudo chmod a+rwx ~/storage

wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
source ~/.bashrc
conda create --name task python=3.8.3
conda activate task
git clone https://github.com/kircore-fm/translation-task
cd translation-task