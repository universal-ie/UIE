# UIE

- Code for [`Unified Structure Generation for Universal Information Extraction`](https://arxiv.org/pdf/2203.12277.pdf)

## Requirements

General

- Python (verified on 3.8)
- CUDA (verified on 11.1/10.2)

Python Packages
CUDA 10.2
``` bash
conda create -n uie python=3.8
conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

CUDA 11.1
``` bash
conda create -n uie python=3.8
pip install -y torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Quick Start

### Data Format

Details of preprocessing see `dataset_processing/`. 
Data folder contains four files：

```text
data/text2spotasoc/absa/14lap
├── entity.schema       # Entity Types for converting SEL to Record
├── relation.schema     # Relation Types for converting SEL to Record
├── event.schema        # Event Types for converting SEL to Record
├── record.schema       # Spot/Asoc Type for constructing SSI
├── test.json
├── train.json
└── val.json
```

train/val/test.json are data files, and each line is a JSON instance.
Each JSON instance contains `text` and `record` fields, in which `text` is plain text, and `record` is the SEL representation of the extraction structure.
Details definition see [DATASETS.md](docs/DATASETS.md).
```text
{
  "text": "Great laptop that offers many great features !",
  "record": "<extra_id_0> <extra_id_0> opinion <extra_id_5> great <extra_id_1> <extra_id_0> aspect <extra_id_5> features <extra_id_0> positive <extra_id_5> great <extra_id_1> <extra_id_1> <extra_id_1>"
}
```

Note:
- Use the extra character of T5 as the structure indicators, such as `<extra_id_0>`, `<extra_id_1>`, `<extra_id_5>`.

| Token  | Role |
| ------------- | ------------- |
| <extra_id_0>  | Start of Label Name |
| <extra_id_1>  | End of Label Name   |
| <extra_id_2>  | Start of Input Text |
| <extra_id_5>  | Start of Text Span  |
| <extra_id_6>  | NULL span for Rejection |

- `record.schema` is the record schema file for building SSI.
It contains three lines: the first line is spot name list, the second line is asoc name list. And the third line is spot-to-asoc dictionary (do not use in code, can be ignored).

  ```text
  ["aspect", "opinion"]
  ["neutral", "positive", "negative"]
  {"aspect": ["neutral", "positive", "negative"], "opinion": []}
  ```

### Pretrained Models
You can find the pre-trained models as following CAS Cloud Box/Google Drive links or download models using command `gdown` (`pip install gdown`).

uie-en-base [[CAS Cloud Box]](https://pan.cstcloud.cn/s/Ncp9mI4Sc0) [[Google Drive]](https://drive.google.com/file/d/12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs/view) [[Huggingface]](https://huggingface.co/luyaojie/uie-base-en)

uie-en-large [[CAS Cloud Box]](https://pan.cstcloud.cn/s/cmZg0iwTok) [[Google Drive]](https://drive.google.com/file/d/15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1/view) [[Huggingface]](https://huggingface.co/luyaojie/uie-large-en)

``` bash
# Example of Google Drive
gdown --id 12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs && unzip uie-base-en.zip
gdown --id 15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1 && unzip uie-large-en.zip
```

Put all models to `hf_models/`.

### Model Training

Training scripts as follows:

- `run_seq2seq.py`: Python code entry
- `run_seq2seq_record.bash`: Model training and evaluating process script.
- `scripts_exp/run_exp.bash`: Model environment configuration and parameter setting entry.

The command for the training is as follows (see bash scripts and Python files for the corresponding command-line
arguments):

```bash
. config/data_conf/base_model_conf_absa.ini  && model_name=uie-base dataset_name=absa/14lap bash scripts_exp/run_exp.bash
```

- `config/data_conf/base_model_conf_absa.ini` refers to using the training settings in `base_model_conf_absa.ini` 
- `model_name=uie-base-en` refers to using uie-base-en.
- `dataset_name=absa/14lap` refers to the dataset path.

Trained models are saved in the `output_dir` specified by `run_seq2seq_record.bash`.

Simple Training Command
```
bash run_seq2seq_record.bash -v -d 2 \
  -b 16 -k 1 --lr 3e-4 --warmup_ratio 0.06
  -i absa/14lap --epoch 50 \
  --spot_noise 0.1 --asoc_noise 0.1 -f spotasoc
  --epoch 50 --map_config config/offset_map/closest_offset_en.yaml
  -m hf_models/uie-base-en \
  --random_prompt
```

Progress logs
```
...
***** Running training *****
  Num examples = 906
  Num Epochs = 50
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 2850
  Num examples = 219
  Batch size = 64
...
```

Final Result
```
...
test offset-rel-strict-P 67.01461377870564
test offset-rel-strict-R 59.11602209944752
test offset-rel-strict-F1 62.81800391389433
...
```

| Metric      | Definition |
| ----------- | ----------- |
| ent-(P/R/F1)      | Micro-F1 of Entity (Entity Type, Entity Span) |
| rel-strict-(P/R/F1)   | Micro-F1 of Relation Strict (Relation Type, Arg1 Span, Arg1 Type, Arg2 Span, Arg2 Type) |
| rel-boundary-(P/R/F1)   | Micro-F1 of Relation Boundary (Relation Type, Arg1 Span, Arg2 Span) |
| evt-trigger-(P/R/F1)   | Micro-F1 of Event Trigger (Event Type, Trigger Span) |
| evt-role-(P/R/F1)   | Micro-F1 of Relation Boundary (Event Type, Arg Role, Arg Span) |


### Model Evaluation

To verify the performance of the UIE requires converting the generated **SEL** expression into **Record** and then evaluating it.


#### 1. Convert structured expressions to record structures (sel2record.py)
After training, `pred_folder` will contain 'eval_preds_seq2seq.txt' or 'test_preds_seq2seq.txt'

``` text
 $ python scripts/sel2record.py -h     
usage: sel2record.py [-h] [-g GOLD_FOLDER] [-p PRED_FOLDER [PRED_FOLDER ...]] [-c MAP_CONFIG] [-d DECODING] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD_FOLDER        folder of golden answer
  -p PRED_FOLDER [PRED_FOLDER ...]
                        multiple different prediction folders
  -c MAP_CONFIG, --config MAP_CONFIG
                        offset matching strategy configuration file, more configuration files are placed in config/offset_map
  -d DECODING           specify structure parser, default is SpotAsoc structure
  -v, --verbose         print more detailed log information
```

#### 2. Validate model performance (eval_extraction.py)
After converting, `pred_folder` will contain 'eval_preds_record.txt' or 'test_preds_record.txt'

```text
 $ python scripts/eval_extraction.py -h   
usage: eval_extraction.py [-h] [-g GOLD_FOLDER] [-p PRED_FOLDER [PRED_FOLDER ...]] [-v] [-w] [-m] [-case]

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD_FOLDER        Golden Dataset folder
  -p PRED_FOLDER [PRED_FOLDER ...]
                        Predicted model folder
  -v                    Show more information during running
  -w                    Write evaluation results to predicted folder
  -m                    Refers to the matching policy
  -case                 Show case study
```

#### 3. Verify the performance of the mapping label (check_offset_map_gold_as_pred.bash)
To verify the effect of structure parser, we took the golden answer `SEL` as the prediction result, and evaluate its performance.
``` bash
bash scripts/check_offset_map_gold_as_pred.bash <data-folder> <map-config>
```

### Data Collator 
**_Sampling Strategy_** and **_Rejection Mechanism_** can be adopted in the training process. 

- `uie/seq2seq/data_collator/meta_data_collator.py` class _DataCollatorForMetaSeq2Seq_ is for collating data, class _DynamicSSIGenerator_ is for prompt sampling
- `run_seq2seq.py` class _DataTrainingArguments_ contains related parameters

Related parameters in class _DataTrainingArguments_ are briefly introduced here: 

- About **_Sampling Strategy_**
``` text
    - max_prefix_length       Maximum length of prompt
    - record_schema           record schema read from record.schema
    - meta_negative           number of negative schema
    - meta_positive_rate      rate of positive spot
    - ordered_prompt          Whether to sort the spot prompt and asoc prompt or not
``` 

- About **_Rejection Mechanism_**
``` text
  - spot_noise              The noise rate of null spot
  - asoc_noise              The noise rate of null asoc
```

## Citation

If this repository helps you, please cite this paper:

Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, Hua Wu.
Unified Structure Generation for Universal Information Extraction.
Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics.

```
@misc{lu-etal-2022-uie,
  author = {Lu, Yaojie and
    Liu, Qing and
    Dai, Dai and
    Xiao, Xinyan and
    Lin, Hongyu and
    Han, Xianpei and
    Sun, Le and
    Wu, Hua},
  title = {Unified Structure Generation for Universal Information Extraction},
  url = {https://arxiv.org/abs/2203.12277},
  year = {2022},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
}
```

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only.
Any commercial use should get formal permission first.