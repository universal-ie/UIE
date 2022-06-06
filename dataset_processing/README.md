# Preprocessing Code for Universal IE Datasets

- Preprocessing code for [`Unified Structure Generation for Universal Information Extraction`](https://arxiv.org/abs/2203.12277)
- Please contact Yaojie Lu ([@luyaojie](mailto:yaojie2017@iscas.ac.cn)) for questions and suggestions.

## Quick Start

1. Prepare datasets following [Dataset preprocessing](#preprocessing)
2. Run `bash run_data_generation.bash` for generating all datasets
3. Run `bash run_sample.bash` for sampling low-resource datasets

## Structure
```
.
├── converted_data/   # Final converted datasets
├── data/             # Raw data
├── data_config/      # Dataset config
├── README.md
├── run_data_generation.bash  # Convert all datasets
├── run_sample.bash           # Sample low-resource datasets
├── scripts/                  # Scripts for preprocessing
├── uie_convert.py            # Main Python File
└── universal_ie/             # Code for preprocessing
```

## <span id="preprocessing">Dataset preprocessing</span>

We follows the preprocessing methods with the following works.

We sincerely thank previous works.

 | Dataset      | Preprocessing |
 | ----------- | ----------- |
 | ACE04 | [mrc-for-flat-nested-ner](https://github.com/ShannonAI/mrc-for-flat-nested-ner) |
 | ACE05 | [mrc-for-flat-nested-ner](https://github.com/ShannonAI/mrc-for-flat-nested-ner) |
 | ACE05-Rel | [sincere](https://github.com/btaille/sincere) |
 | CoNLL 04 | [sincere](https://github.com/btaille/sincere) |
 | NYT | [JointER](https://github.com/yubowen-ph/JointER/tree/master/dataset/NYT-multi/data) |
 | SCIERC | [dygiepp](https://github.com/dwadden/dygiepp) |
 | ACE05-Evt | [OneIE](http://blender.cs.illinois.edu/software/oneie/) |
 | CASIE | [CASIE](https://github.com/Ebiquity/CASIE), Our preprocessing code see [here](data/casie/). |
 | 14lap | [BARTABSA](https://github.com/yhcc/BARTABSA/tree/main/data/pengb) |
 | l4res | [BARTABSA](https://github.com/yhcc/BARTABSA/tree/main/data/pengb) |
 | 15res | [BARTABSA](https://github.com/yhcc/BARTABSA/tree/main/data/pengb) |
 | 16res | [BARTABSA](https://github.com/yhcc/BARTABSA/tree/main/data/pengb) |

### ABSA

```
git clone https://github.com/yhcc/BARTABSA data/BARTABSA
mv data/BARTABSA/data data/absa
```

### Entity

``` bash
# CoNLL03
mkdir data/conll03
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train -P data/conll03
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa -P data/conll03
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb -P data/conll03

# gdown >= 4.4.0
pip install -U gdown
mkdir data/mrc_ner
# ACE04
gdown 1U-hGOgLmdqudsRdKIGles1-QrNJ7SSg6 -O data/mrc_ner/ace2004.tar.gz
tar zxvf data/mrc_ner/ace2004.tar.gz -C data/mrc_ner

# ACE05
gdown 1iodaJ92dTAjUWnkMyYm8aLEi5hj3cseY -O data/mrc_ner/ace2005.tar.gz
tar zxvf data/mrc_ner/ace2005.tar.gz -C data/mrc_ner
```

### Relation

#### NYT
``` bash
mkdir data/NYT-multi
wget -P data/NYT-multi https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/train.json
wget -P data/NYT-multi https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/dev.json
wget -P data/NYT-multi https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data/test.json
```

#### CoNLL04/ACE05-rel
1. Use the preprocessing code from [sincere](https://github.com/btaille/sincere) repo, then convert them to same format.
Please follow the instructions to preprocess the CoNLL04 and ACE05-rel datasets, and put them at `data/sincere/`.
```
 $ tree data/sincere/
data/sincere/
├── ace05.json
└── conll04.json
```

2. Convert conll/ace05-rel in sincere
```
python scripts/sincere_processing.py
```

#### SciERC
We first use the preprocessing code for SciERC from [DyGIE](https://github.com/dwadden/dygiepp) repo.
Please put the preprocessed dataset in the `collated_data` of dygiepp to
``` text
$ tree data/dygiepp
data/dygiepp
└── scierc
    ├── dev.json
    ├── test.json
    └── train.json
```
Then convert `scierc` in to relation format as CoNLL04/ACE05-Rel.
``` bash
mkdir -p data/relation/scierc
python scripts/scierc_processing.py
```
### Event

#### ACE05-Evt
The preprocessing code of ACE05-Evt is following [OneIE](http://blender.cs.illinois.edu/software/oneie/).
Please follow the instructions and put preprocessed dataset at `data/oneie`:
``` text
## OneIE Preprocessing, ACE_DATA_FOLDER -> ace_2005_td_v7
$ tree data/oneie/ace05-EN 
data/oneie/ace05-EN
├── dev.oneie.json
├── english.json
├── english.oneie.json
├── test.oneie.json
└── train.oneie.json
```

Note:
- `nltk==3.5` is used in our experiments, we found `nltk==3.6+` may leads different sentence numbers.

#### CASIE

The preprocessing code of casie is inlucluded in the data/casie.
Please follow the instructions as follow:
```
cd data/casie
bash scripts/download_data.bash
bash scripts/download_corenlp.bash
bash run.bash
cd ../../
```

## Conversion detailed steps

1. Read the dataset configuration file and automatically find the task_format class that reads the data
2. Based on the configuration file, Task Format instance reads the data
3. Generate data in the corresponding format according to different generation_formats
  - Reads the label mapping in the configuration file for changing label name from raw annotation
  - Generate data file format

### Example of Dataset Configuration

```yaml
# data_config/entity/conll03.yaml
name: conll03               # Dataset Name
path: data/conll03  # Dataset Folder
data_class: CoNLL03         # Task Format
split:                      # Dataset Split
  train: eng.train
  val: eng.testa
  test: eng.testb
language: en
mapper: # Label Mapper
  LOC: location
  ORG: organization
  PER: person
  MISC: miscellaneous
```

``` text
 $ tree converted_data/text2spotasoc/entity/conll03
converted_data/text2spotasoc/entity/conll03
├── entity.schema
├── event.schema
├── record.schema
├── relation.schema
├── test.json
├── train.json
└── val.json
```

Example of entity
```json
{
  "text": "EU rejects German call to boycott British lamb .",
  "tokens": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
  "record": "<extra_id_0> <extra_id_0> organization <extra_id_5> EU <extra_id_1> <extra_id_0> miscellaneous <extra_id_5> German <extra_id_1> <extra_id_0> miscellan
eous <extra_id_5> British <extra_id_1> <extra_id_1>",
  "entity": [{"type": "miscellaneous", "offset": [2], "text": "German"}, {"type": "miscellaneous", "offset": [6], "text": "British"}, {"type": "organization", "offset": [0], "text": "EU"}],
  "relation": [],
  "event": [],
  "spot": ["organization", "miscellaneous"],
  "asoc": [],
  "spot_asoc": [{"span": "EU", "label": "organization", "asoc": []}, {"span": "German", "label": "miscellaneous", "asoc": []}, {"span": "British", "label": "miscellaneous", "asoc": []}]
}
```

## Low-resource dataset sampling

See details in `run_sample.bash`, it will generate all low-resource datasets for experiments.

```
bash run_sample.bash
```

### Low-ratio smaple

``` text
 $ python scripts/sample_data_ratio.py -h
usage: sample_data_ratio.py [-h] [-src SRC] [-tgt TGT] [-seed SEED]

optional arguments:
  -h, --help  show this help message and exit
  -src SRC
  -tgt TGT
  -seed SEED
```

Usage: sample 0.01/0.05/0.1 of training instances for low-ratio experiments

``` bash
python scripts/sample_data_ratio.py \
  -src converted_data/text2spotasoc/entity/mrc_conll03 \
  -tgt test_conll03_ratio 
```

### N-shot Sample

``` text
 $ python scripts/sample_data_shot.py -h
usage: sample_data_shot.py [-h] -src SRC -tgt TGT -task {entity,relation,event} [-seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -src SRC              Source Folder Name
  -tgt TGT              Target Folder Name, n shot sampled
  -task {entity,relation,event}
                        N-Shot Task name
  -seed SEED            Default is None, no random
```

Usage: sample 1/5-10-shot of training instances for low-shot experiments


``` bash
python scripts/sample_data_shot.py \
  -src converted_data/text2spotasoc/entity/mrc_conll03 \
  -tgt test_conll03_shot \
  -task entity
```

Note:
- `-task` indicates the target task: `entity`, `relation` and `event`
