# ICASSP 2024 - LEARNING FROM EASY TO HARD: MULTI-TASK LEARNING WITH DATA SCHEDULING
Multi-task learning with data scheduler (assigns larger weights to easy samples during early training stage and gradually treat all samples equally)
## SLDS
![SLDS Diagram](https://github.com/ZeyuLiu0706/SLDS/blob/main/img/SLDS.png)
## Quick Start 
```python
python main_fed.py -algo fedgr/fednova/fedavg/fedopt/moon -dataset cifar10/cifar100/fashion-mnist
```
## Citation
This is the code for the 2023 DASFAA paper: FedGR: Federated Learning with Gravitation Regulation for Double Imbalance Distribution.
**Please cite our paper if you use the code:**
```
@inproceedings{Guo2023FedGR
  author    = {Songyue Guo and
               Xu Yang and
               Jiyuan Feng and
               Ye Ding and 
               Wei Wang and
               Yunqing Feng and
               Qing Liao},
  title     = {FedGR: Federated Learning with Gravitation Regulation for Double Imbalance Distribution
},
  booktitle = {Database Systems for Advanced Applications - 28th International Conference,
               {DASFAA} 2023, Tianjin, China, April 17-20, 2023},
  publisher = {Springer},
  year      = {2023}
}
```
