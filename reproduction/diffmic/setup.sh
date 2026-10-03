#!/bin/bash
# One-time setup on a Slurm cluster (run on a login node):
#   cd ~/leakage && bash setup.sh
# Adjust the module name below to your cluster's conda/miniconda module.
set -e
cd "$HOME/leakage"
module load miniconda3/24.1.2-py310
if ! conda env list | grep -q "^diffmic "; then conda create -y -q -n diffmic python=3.11; fi
source activate diffmic
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q pyyaml pandas scikit-learn scipy statsmodels scikit-image imageio opencv-python-headless tensorboard kaggle
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"

# Kaggle key: upload kaggle.json into ~/leakage first
if [ -f kaggle.json ]; then mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json; fi
if [ ! -d aptos/train_images ] || [ "$(ls aptos/train_images | wc -l)" -lt 3662 ]; then
  [ -f ~/.kaggle/kaggle.json ] || { echo "STOP: upload kaggle.json into the leakage folder, then run this again"; exit 1; }
  kaggle competitions download -c aptos2019-blindness-detection -p .
  mkdir -p aptos && cd aptos && unzip -q -o ../aptos2019-blindness-detection.zip 'train.csv' 'train_images/*' && cd ..
  rm -f aptos2019-blindness-detection.zip
fi
echo "APTOS training images: $(ls aptos/train_images | wc -l)  (should be 3662)"
echo "SETUP OK - now run:  sbatch phase5.sbatch"
