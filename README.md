# Code and Appendix of AAAI 2023 paper:  

FedMDFG: Federated Learning with Multi-Gradient Descent and Fair Guidance



[![Gitter](https://badges.gitter.im/Federated-Learning-Discussion/community.svg)](https://gitter.im/Federated-Learning-Discussion/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)




A more powerful platform is released in another repository: https://github.com/zibinpan/FedLF, which includes more SOTA fair FL algorithms.

# 用法
```
python run.py + 参数
```

# 关键文件
[FedMDFG算法](/fedplat/algorithm/FedMDFG/FedMDFG.py)(修改部分：169行) train_a_round函数为训练主函数  
[主函数](/fedplat/main.py)  
[模型](/fedplat/model/CNN_OriginalFedAvg.py)  
[数据集](/fedplat/dataloaders/DataLoader_cifar10.py)  

# 关键命令行参数
命令行参数在main.py中  
seed 随机数种子  
device 设备  
model 模型  
algorithm 算法  
dataloader 数据集  
N 客户端数量  
B batch size每个批次的大小  
C 每个客户被选择的概率，1为使用所有客户模型  
R 全局训练次数  
E 客户端本地训练次数  
lr 学习率  
decay 学习率下降率  
推荐参数：
```
python run.py --seed 1 --device 0 --model CNN_CIFAR10_FedAvg --algorithm FedMDFG --dataloader DataLoader_cifar10 --N 5 --B 50 --C 1 --R 3000 --E 3 --lr 0.1 --decay 0.999
```

# 改进算法思想
对于每个用户模型的loss与对应引导向量值成正比
