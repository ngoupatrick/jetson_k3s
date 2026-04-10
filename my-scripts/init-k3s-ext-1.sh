#!/bin/bash

# make sure to run this script from the root of the project
# mkdir -p ~/my-scripts
# vi ~/my-scripts/init-k3s-ext-1.sh
# chmod +x ~/my-scripts/init-k3s-ext-1.sh
# echo "picocluster" > ~/my-scripts/.pass && chmod 600 ~/my-scripts/.pass

echo "== Initializing /k3s-ext-1 folder == \n"

echo "== 1. Downloading weights models file for resnet50, mobilenet_v3_large, efficientnet_v2_m and convnext_tiny ... == \n"
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
wget https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth
wget https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth
wget https://download.pytorch.org/models/convnext_tiny-983f1562.pth

echo "== 2. Creating directories /k3s-ext-1/code/scripts and /k3s-ext-1/code/models ... == \n"
cat ~/my-scripts/.pass | sudo -S mkdir -p /k3s-ext-1/code/scripts
cat ~/my-scripts/.pass | sudo -S mkdir -p /k3s-ext-1/code/models

echo "== 3. Moving model files to /k3s-ext-1/code/models ... == \n"

echo "== 3.1 Moving resnet50-0676ba61.pth ... == \n"
cat ~/my-scripts/.pass | sudo -S mv resnet50-0676ba61.pth /k3s-ext-1/code/models

echo "== 3.2 Moving mobilenet_v3_large-5c1a4163.pth ... == \n"
cat ~/my-scripts/.pass | sudo -S mv mobilenet_v3_large-5c1a4163.pth /k3s-ext-1/code/models

echo "== 3.3 Moving efficientnet_v2_m-dc08266a.pth ... == \n"
cat ~/my-scripts/.pass | sudo -S mv efficientnet_v2_m-dc08266a.pth /k3s-ext-1/code/models

echo "== 3.4 Moving convnext_tiny-983f1562.pth ... == \n"
cat ~/my-scripts/.pass | sudo -S mv convnext_tiny-983f1562.pth /k3s-ext-1/code/models

echo "== 4. End of script ... == \n"