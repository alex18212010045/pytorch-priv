cd /workspace/mnt/group/general-reg/yangmingzhao/2019.1.3cifardrop/pytorch-priv/
pip2 install --index https://pypi.mirrors.ustc.edu.cn/simple/ --upgrade pip
pip2 install --index https://pypi.mirrors.ustc.edu.cn/simple/ easydict
pip2 install --index https://pypi.mirrors.ustc.edu.cn/simple/ pyyaml
pip install tensorboardX
python2 tools/cls_dropblock_imagenet.py --cfg cfgs/imagenet/resdropnet50_1x64d-Copy1.yml