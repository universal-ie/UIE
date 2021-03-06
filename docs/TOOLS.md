# Tools for UIE

### Evaluate Model Performance
验证模型性能 (eval_extraction.py)
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
  -m                    Match predicted result multiple times
  -case                 Show case study
```

### Check Offset Mapping Performance
验证回标的性能 (check_offset_map_gold_as_pred.bash)
``` bash
bash scripts/check_offset_map_gold_as_pred.bash <data-folder> <map-config>
```

### Convert SEL to Record
将结构化表达式转换成 Record 结构 (sel2record.py)
``` text
 $ python scripts/sel2record.py -h     
usage: sel2record.py [-h] [-g GOLD_FOLDER] [-p PRED_FOLDER [PRED_FOLDER ...]] [-c MAP_CONFIG] [-d DECODING] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -g GOLD_FOLDER        标准答案（Gold）文件夹
  -p PRED_FOLDER [PRED_FOLDER ...]
                        多个不同的预测（Pred）文件夹
  -c MAP_CONFIG, --config MAP_CONFIG
                        Offset 匹配策略的配置文件
  -d DECODING           使用 SpotAsoc 结构的解析器进行结构表达式解析
  -v, --verbose         打印更详细的日志信息
```
