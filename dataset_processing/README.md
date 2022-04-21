# Universal IE

## Structure

1. universal_ie.generation_format: Target Format for data
2. universal_ie.task_format: Source Format for data loading
3. universal_ie.ie_format: IE Element format
4. converted_data: Output Folder
5. data: Raw Data Folder
6. data_config: Data Config

## Main scripts

1. uie_convert.py: Main Python Entry
2. run_data_generation.bash: Generate all datasets
3. run_sample.bash: Generate all low-resource datasets [Details](run_sample.md)

## Convertion 

1. Read the configuration file and automatically find the task_format class that reads the data
2. Based on the configuration file, the Task Format instance reads the data
3. Generate data in the corresponding format according to different generation_formats
  - Reads the label mapping in the configuration file  
  - Generate data file format  

``` text
 $ tree converted_data/text2spotasoc/event/oneie_ace05_en_event
converted_data/text2spotasoc/event/oneie_ace05_en_event
├── entity.schema       # Entity Types for converting SEL to Record
├── relation.schema     # Relation Types for converting SEL to Record
├── event.schema        # Event Types for converting SEL to Record
├── record.schema       # Spot/Asoc Type for constructing SSI
├── test.json
├── train.json
└── val.json
```

```json
{
    "text": "Next week ’ s trial of Mee is expected to attract widespread media attention .",
    "tokens": ["Next", "week", "’", "s", "trial", "of", "Mee", "is", "expected", "to", "attract", "widespread", "media", "attention", "."],
    "record": "<extra_id_0> <extra_id_0> Trial-Hearing <extra_id_5> trial <extra_id_0> Defendant <extra_id_5> Mee <extra_id_1> <extra_id_1> <extra_id_0> Person <extra_id_5> Mee <extra_id_1> <extra_id_1>",
    "entity": [{"type": "Person", "offset": [6], "text": "Mee"}], 
    "relation": [],
    "event": [{"type": "Trial-Hearing", "offset": [4], "text": "trial", "args": [{"type": "Defendant", "offset": [6], "text": "Mee"}]}],
    "spot": ["Trial-Hearing", "Person"],
    "asoc": ["Defendant"],
    "spot_asoc": [{"span": "trial", "label": "Trial-Hearing", "asoc": [["Defendant", "Mee"]]}]
}
```

## Dataset Config

```yaml
# data_config/entity/ace2005_english_name.yaml
name: ace2005_english_name                          # Dataset Name
path: data/spannet_data/entity/ace2005_english_name # Dataset Folder
data_class: Spannet                                 # Task Format
split:                                              # Dataset Split
  train: train.jsonlines
  val: dev.jsonlines
  test: test.jsonlines
language: en

mapper:
  FAC: facility
  GPE: geographical social political
  LOC: location
  ORG: organization
  PER: person
  VEH: vehicle
  WEA: weapon
```
