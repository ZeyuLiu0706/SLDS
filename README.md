# ICASSP 2024 - LEARNING FROM EASY TO HARD: MULTI-TASK LEARNING WITH DATA SCHEDULING
MTL with data scheduler (assigns larger weights to easy samples during early training stage and gradually treat all samples equally)
## SLDS
![SLDS Diagram](https://github.com/ZeyuLiu0706/SLDS/blob/main/img/SLDS.png)
## Quick Start 
```python
python NYUv2_SLDS.py --data_path ./data/nyuv2
```

## Datasets

The `MultiMNIST` can be found [here](https://github.com/intel-isl/MultiObjectiveOptimization).

The `NYUv2` can be found [here](https://drive.google.com/file/d/11pWuQXMFBNMIIB4VYMzi9RPE-nMOBU8g/view).


## Citation
This is the code for the 2024 ICASSP paper: LEARNING FROM EASY TO HARD: MULTI-TASK LEARNING WITH DATA SCHEDULING.

**Please cite our paper if you use the code:**
```
@inproceedings{Liu2024SLDS
  author    = {Zeyu Liu and
               Heyan Chai and
               Qing Liao},
  title     = {LEARNING FROM EASY TO HARD: MULTI-TASK LEARNING WITH DATA SCHEDULING},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```
