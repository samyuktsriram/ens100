# ens100
for the ENS Challenge, DSB 2024-26

for large files, make a folder called data_large_files/ and add whatever you need there on your local. that'll be the path we can commonly reference for input / output / model saving

TO DO LIST

1. 

Read

CNN
- https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
- https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939

Residuals Neural Networks
- https://medium.com/@ibtedaazeem/understanding-resnet-architecture-a-deep-dive-into-residual-neural-network-2c792e6537a9
- https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624
- https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c
- https://towardsdatascience.com/intuition-behind-residual-neural-networks-fa5d2996b2c7

Wide Residual Netwroks
- https://arxiv.org/pdf/1605.07146
  
Models:
- https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4


Python Environment Setup
```console
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installing, close and reopen your terminal application or refresh it by running the following command:

```console
source ~/miniconda3/bin/activate
conda init --all
```

Create Environment:

```console
conda create -n erwan python=3.10
conda activate erwan
```

Install packages:

```console
conda create -n erwan python=3.10
conda activate erwan
```

**To Run Jupyter Notebook (don't you dare look at this Sam)**

```console
pip install notebook
```

Not sure we need this (???)
---
```console
sudo apt-get install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" |sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
rm -f packages.microsoft.gpg
```

```console
sudo apt install apt-transport-https
sudo apt update
sudo apt install code # or code-insiders
```
---

Make sure to install the python and jupyter extensions in vs code, just go to the extentions tab
Select the Kernel after this.

**Install packages**

```console
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn scipy tqdm pillow matplotlib scikit-image pandas
```














