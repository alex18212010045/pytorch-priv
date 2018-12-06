cd /workspace/mnt/group/terror/yangmingzhao/2018.12.6githubissues/pytorch-priv/
pip install --index https://pypi.mirrors.ustc.edu.cn/simple/ --upgrade pip
pip install --index https://pypi.mirrors.ustc.edu.cn/simple/ easydict
pip install --index https://pypi.mirrors.ustc.edu.cn/simple/ pyyaml
pip install dropblock
pip install tensorboardX
python3 cls_train.py --cfg cfgs/imagenet/resnet50_1x64d.yml