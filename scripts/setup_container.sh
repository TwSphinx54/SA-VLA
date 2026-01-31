#!/bin/bash
set -e

echo "=== Backing up existing sources.list and sources.list.d directory ==="
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
sudo mkdir -p /etc/apt/sources.list.d/backup
sudo mv /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/backup/ || true

echo "=== Writing new sources.list (Tsinghua mirror) ==="
sudo bash -c 'cat > /etc/apt/sources.list <<EOF
# Ubuntu Tsinghua mirror
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
EOF'

echo "=== Removing all possible Nvidia CUDA source files ==="
sudo find /etc/apt/sources.list.d/ -type f -iname "*cuda*.list" -exec rm -v {} \; || true

echo "=== Cleaning APT cache and updating index ==="
sudo apt-get clean
sudo apt-get update

echo "=== Checking if USTC or NVIDIA official sources still exist ==="
if grep -R "ustc\.edu\.cn\|developer\.download\.nvidia\.com" /etc/apt/sources.list /etc/apt/sources.list.d/ ; then
  echo "⚠️ Warning: USTC or NVIDIA official sources still detected, please check manually!"
else
  echo "✅ All sources replaced with Tsinghua mirror, only Tsinghua sources are in use."
fi

echo "=== Completed ==="

uv pip install wand scikit-image

sudo apt update
sudo apt install -y imagemagick libmagickwand-dev