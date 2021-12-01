# occupancy_prediction
CMSC828I Course Project

Download the data from this link: [AI2THOR OccMap Data](https://obj.umiacs.umd.edu/shareddata/AI2THOROccMapData.zip)

After unzipping, the `inp_data`, `gt_data` directories and `description_ang0.csv` should be in the same directory as train.py

Example run command:
```
python -u train.py --epochs 50 --batch-size 16 --learning-rate 0.01 --validation 10
``` 
