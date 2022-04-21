# Universal IE

通用生成式 IE 转换脚本

## 文件夹说明

1. universal_ie.generation_format 生成目标格式，用于生成对应生成格式
2. universal_ie.task_format 原始IE任务数据格式，用于读取数据
3. universal_ie.ie_format 核心 IE 元素的定义
4. converted_data 生成格式的数据
5. data 原始格式的数据
6. data_config 数据集配置文件

## 转换脚本入口

1. uie_convert.py: 数据转换入口
2. convert_wiki_entity.py: Wiki自动标注数据入口
3. run_data_generation.bash: 生成所有数据
4. run_sample.bash: 生成所有低资源数据，[说明](run_sample.md)

## 转换脚本的通用逻辑

1. 读取配置文件，自动找到读取数据的 task_format 类
2. 依据配置文件，task_format 实例对数据进行读取
3. 依据不同的 generation_format 进行对应格式的数据生成
    - 读取配置文件中的标签映射关系
    - 生成数据文件格式

```text
生成的数据文件夹
 $ tree converted_data/text2spotasoc/event/oneie_ace05_en/
converted_data/text2spotasoc/event/oneie_ace05_en/
├── entity.schema    # 实体类别信息，第一行是实体列表
├── event.schema     # 事件类别信息，第一行是事件类别列表，第二行是事件论元列表
├── record.schema    # 记录信息，第一行是 spot 类别列表，第二行是 asoc 类别列表，第三行是 spot-asoc 关联关系
├── relation.schema  # 关系类别信息，第一行是关系列表列表，第二行是关系论元列表
├── test.json        # 测试数据，格式如下
├── train.json       # 训练数据，格式如下
└── val.json         # 验证数据，格式如下
```

```json
{
    "text": "Next week ’ s trial of Mee is expected to attract widespread media attention .",
    "tokens": ["Next", "week", "’", "s", "trial", "of", "Mee", "is", "expected", "to", "attract", "widespread", "media", "attention", "."],
    "record": "<extra_id_0> <extra_id_0> Trial-Hearing <extra_id_5> trial <extra_id_0> Defendant <extra_id_5> Mee <extra_id_1> <extra_id_1> <extra_id_0> Person <extra_id_5> Mee <extra_id_1> <extra_id_1>",
    "entity_offsets": [{"type": "Person", "offset": [6], "text": "Mee"}], 
    "relation_offsets": [],
    "event_offsets": [{"type": "Trial-Hearing", "offset": [4], "text": "trial", "args": [{"type": "Defendant", "offset": [6], "text": "Mee"}]}],
    "spot": ["Trial-Hearing", "Person"],
    "asoc": ["Defendant"],
    "spot_asoc": [{"span": "trial", "label": "Trial-Hearing", "asoc": [["Defendant", "Mee"]]}]
}
```

## 配置文件说明

```yaml
# data_config/entity/ace2005_english_name.yaml
name: ace2005_english_name                          # 数据名称
path: data/spannet_data/entity/ace2005_english_name # 数据文件夹路径
data_class: Spannet                                 # 读取该数据所使用的 task_format clas
split:                                              # 数据划分方式
  train: train.jsonlines                            # 数据文件夹中训练集文件
  val: dev.jsonlines                                # 数据文件夹中开发集文件
  test: test.jsonlines                              # 数据文件夹中测试集文件
language: en                                        # 数据集的语言

mapper:
  FAC: facility
  GPE: geographical social political
  LOC: location
  ORG: organization
  PER: person
  VEH: vehicle
  WEA: weapon
```
