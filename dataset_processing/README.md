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

Example of event
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

Schema file
```
# event.schema / same as record.schema
["elect", "born", "transport", "phone write", "marry", "die", "start position", "injure", "transfer ownership", "meet", "nominate", "attack", "start organization", "trial hearing", "convict", "sentence", "end position", "divorce", "acquit", "charge indict", "transfer money", "appeal", "sue", "merge organization", "declare bankruptcy", "execute", "arrest jail", "extradite", "demonstrate", "end organization", "release parole", "fine", "pardon"]
["recipient", "victim", "target", "entity", "beneficiary", "plaintiff", "agent", "adjudicator", "place", "seller", "destination", "instrument", "artifact", "vehicle", "origin", "prosecutor", "organization", "attacker", "defendant", "buyer", "giver", "person"]
{"elect": ["entity", "place", "person"], "born": ["place", "person"], "transport": ["artifact", "vehicle", "victim", "origin", "agent", "place", "destination"], "phone write": ["place", "entity"], "marry": ["place", "person"], "die": ["victim", "agent", "place", "instrument", "person"], "start position": ["entity", "place", "person"], "injure": ["instrument", "victim", "agent", "place"], "transfer ownership": ["artifact", "beneficiary", "place", "seller", "buyer"], "meet": ["place", "entity"], "nominate": ["agent", "person"], "attack": ["victim", "target", "agent", "attacker", "place", "instrument"], "start organization": ["organization", "place", "agent"], "trial hearing": ["adjudicator", "prosecutor", "place", "defendant"], "convict": ["place", "adjudicator", "defendant"], "sentence": ["place", "adjudicator", "defendant"], "end position": ["entity", "place", "person"], "divorce": ["place", "person"], "acquit": ["adjudicator", "defendant"], "charge indict": ["prosecutor", "place", "adjudicator", "defendant"], "transfer money": ["recipient", "giver", "place", "beneficiary"], "appeal": ["plaintiff", "place", "adjudicator"], "sue": ["plaintiff", "place", "adjudicator", "defendant"], "merge organization": ["organization"], "declare bankruptcy": ["organization", "place"], "execute": ["agent", "place", "person"], "arrest jail": ["agent", "place", "person"], "extradite": ["person", "destination", "agent", "origin"], "demonstrate": ["place", "entity"], "end organization": ["organization", "place"], "release parole": ["entity", "place", "person"], "fine": ["place", "adjudicator", "entity"], "pardon": ["place", "adjudicator", "defendant"]}
```

## Dataset Config

```yaml
# data_config/entity/conll03.yaml
name: conll03               # Dataset Name
path: data/conll03/conll03  # Dataset Folder
data_class: CoNLL03         # Task Format
split:                      # Dataset Split
  train: eng.train
  val: eng.testa
  test: eng.testb
language: en
mapper:
  LOC: location
  ORG: organization
  PER: person
  MISC: miscellaneous
```
