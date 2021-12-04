# occupancy_prediction
CMSC828I Course Project

Download the data from this link: [AI2THOR OccMap Data](https://obj.umiacs.umd.edu/shareddata/AI2THOROccMapData.zip)

After unzipping, the `inp_data`, `gt_data` directories and `description_ang0.csv` should be in the same directory as train.py

Example run command:
- Training
```
python train.py --epochs 100 --batch-size 16 --learning-rate 0.01 --validation 10 --loss-function 'kl_raw' --scale 1
```
- Testing
```
python test.py --model_path saved_models/sgd_LR_0.01_epoch_500_mse_scale_5.0_PROB_ODDSCLAE10.pth --batch-size 16 --show
``` 
