cd /workspace/mnt/group/ocr-fd-group/yangmingzhao/2018.12.18dropblock/pytorch-priv/
pip2 install --index https://pypi.mirrors.ustc.edu.cn/simple/ --upgrade pip
pip2 install --index https://pypi.mirrors.ustc.edu.cn/simple/ easydict
pip2 install --index https://pypi.mirrors.ustc.edu.cn/simple/ pyyaml
pip install tensorboardX
python2 tools/cls_transblock_imagenet.py --cfg cfgs/imagenet/restransnet50_1x64d.yml