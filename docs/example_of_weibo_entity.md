# An example of Chinese Weibo Entity Extraction

## Data Preprocessing
``` bash
cd dataset_processing/
git clone https://github.com/hltcoe/golden-horse data/golden-horse
python scripts/preprocess_golden_horse.py
python uie_convert.py -config data_config/entity_zh/zh_weibo.yaml -output entity_zh
cd ..
```

It also need to link conveted data folder to `data`.

``` bash
 $ tree data/text2spotasoc/entity_zh/zh_weibo/
data/text2spotasoc/entity_zh/zh_weibo/
├── entity.schema
├── event.schema
├── record.schema
├── relation.schema
├── test.json
├── train.json
└── val.json
```

## Download checkpoint

uie-char-small (chinese) [[CAS Cloud Box]](https://pan.cstcloud.cn/s/J7HOsDHHQHY)

## Run Example

``` bash
bash run_uie_finetune.bash -v -d 2 \
  -b 16 -k 1 --lr 1e-4 --warmup_ratio 0.06 \
  -i entity_zh/zh_weibo --epoch 50 \
  --spot_noise 0.1 --asoc_noise 0.1 -f spotasoc \
  --epoch 50 \
  --map_config config/offset_map/longer_first_offset_zh.yaml \
  -m hf_models/uie-char-small \
  --random_prompt
```

Progress logs
```
...
***** Running training *****| 0/1 [00:00<?, ?ba/s]
  Num examples = 1350
  Num Epochs = 50
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 4250
...
```
