## Dataset
The dataset for this project comes from the open source platform: https://electrosense.org

| Dataset parameters            | Value                                    |
|-------------------------------|------------------------------------------|
| Dataset                       | Spectrum measurement data of ElectroSense |
| Dataset source                | https://electrosense.org                 |
| Sensor location               | Madrid, Spain                            |
| Frequency band                | 600 MHz–700 MHz                          |
| Monitoring time               | 6/1/2021 – 6/8/2021                      |
| Frequency resolution          | 2 MHz                                    |
| Time resolution               | 1 minutes                                |

**！！！The opening time of sensors on this platform is uncertain, and there may be some sensors shutdown.**

## Usage

### Requirements
- Numpy
- pytorch
- pytorch-lightning
- pandas
- scipy
- matplotlib

### Model training, vali and test
```
python train_main.py bcn_L --gpus 1
```
where "bcn_L" can be replaced by "rack_2", "test_rpi4" or "test_yago".
### Model test
```
python test_main.py --model_name GSLGCRNN --max_epochs 3000 --learning_rate 0.0001 --batch_size 32 --hidden_dim 64  --settings supervised --gpus 1
```


Run `tensorboard --logdir lightning_logs/version_0 --samples_per_plugin scalars=999999999` in terminal to view the prediction results and experimental indicators.

## Acknowledgement
Please acknowledge the following paper if the codes is useful for your research.

```
@INPROCEEDINGS{10758056,
  author={Yang, Dong and Wang, Yue and Cai, Zhipeng and Li, Yingshu},
  booktitle={2024 IEEE 100th Vehicular Technology Conference (VTC2024-Fall)}, 
  title={Spectrum Prediction via Graph Structure Learning}, 
  year={2024},
  pages={1-5},
  doi={10.1109/VTC2024-Fall63153.2024.10758056}}
  ```
