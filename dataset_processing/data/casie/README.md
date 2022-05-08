# Preprocessing code for CASIE

## Dataset Download
Download the raw data from [CASIE](https://github.com/Ebiquity/CASIE).

```bash
bash scripts/download_data.bash
bash scripts/download_corenlp.bash
```

## Dataset Preprocessing

```bash
bash run.bash
# python scripts/generate_content.py raw_data/annotation raw_data/content
# bash scripts/run_stanford_corenlp.bash
# python scripts/extract_doc_json.py
# python scripts/split_data.py
```

Details of preprocessing
- Excluding three documents: 10001, 10002 and 999
- Run Corenlp to preprocessing: sentence split
- Split Train/Dev/Test according the date of document

The above scripts will generate `data` folder.

```text
data
├── dev.jsonlines
├── test.jsonlines
└── train.jsonlines
```

## Data Format

```text
# doc
{
  "id": "XIN20001201.2000.0146",  # Document id
  "sentences": [sentence0, sentence1, ...],
  "event": [event0, event1, ...],
  "text": text,
  "info": {
    "title": text,
    "date": YYYY_MM_DD,
    "type": 'text',
    "link": url,
  }
}

# sentence
{
  "tokens": [token0, token1, ...],
  "span": (start, end), # Offset boundary of sentence, end is excluding.
}

# token in corenlp format
{
  "characterOffsetBegin": int,
  "characterOffsetEnd": int, # Offset boundary of token, the end position is excluding.
  "word": "",
  "originalText": "",
}

# event
{
  "id": "",
  "mentions": [{
    "id": "",
    "type": "",
    "subtype": "",
    "nugget": {
      "text": "",
      "span": (start, end),  # Boundary of span, the end position is including.
      "tokens": [(sentid, tokenid), ...]
    },
    "arguments": [{
      "id": "",
      "role": "",
      "filler_type": "",
      "text": "",
      "span": (start, end),   # Boundary of span, the end position is including.
      "tokens": [(sentid, tokenid), ...]
      }, argument_mention1, ...]
    },
   mention1, ...],
}
```
