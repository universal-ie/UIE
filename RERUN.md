ABSA UIE-BASE
``` bash
. config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/14lap bash scripts_exp/run_exp.bash

. config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/15res bash scripts_exp/run_exp.bash

. config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/16res bash scripts_exp/run_exp.bash

. config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/14res bash scripts_exp/run_exp.bash
```

ABSA UIE-large
``` bash
. config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/14lap bash scripts_exp/run_exp.bash

. config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/15res bash scripts_exp/run_exp.bash

. config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/16res bash scripts_exp/run_exp.bash

. config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/14res bash scripts_exp/run_exp.bash
```

UIE-base Ratio
``` bash
. config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=entity/conll03 bash scripts_exp/run_exp_ratio.bash

. config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=relation/conll04 bash scripts_exp/run_exp_ratio.bash

. config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=event/oneie_ace05_en_event bash scripts_exp/run_exp_ratio.bash

. config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=absa/16res bash scripts_exp/run_exp_ratio.bash
```

UIE base Shot
```bash
. config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=entity/conll03 bash scripts_exp/run_exp_shot.bash

. config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=relation/conll04 bash scripts_exp/run_exp_shot.bash

. config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=event/oneie_ace05_en_event bash scripts_exp/run_exp_shot.bash

. config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=absa/16res bash scripts_exp/run_exp_shot.bash
```

UIE Large Entity
```bash
. config/data_conf/large_conll03_conf.ini && model_name=uie-large-en dataset_name=entity/conll03 bash scripts_exp/run_exp.bash

. config/data_conf/large_ace04ent_conf.ini && model_name=uie-large-en dataset_name=entity/mrc_ace04 bash scripts_exp/run_exp.bash

. config/data_conf/large_ace05ent_conf.ini && model_name=uie-large-en dataset_name=entity/mrc_ace05 bash scripts_exp/run_exp.bash
```

UIE Large Relation
```bash
. config/data_conf/large_scierc_conf.ini && model_name=uie-large-en dataset_name=relation/scierc bash scripts_exp/run_exp.bash

. config/data_conf/large_conll04_conf.ini && model_name=uie-large-en dataset_name=relation/conll04 bash scripts_exp/run_exp.bash

. config/data_conf/large_ace05rel_conf.ini && model_name=uie-large-en dataset_name=relation/ace05-rel bash scripts_exp/run_exp.bash
```

UIE Large Event
```bash
. config/data_conf/large_ace05evt_conf.ini && model_name=uie-large-en dataset_name=event/oneie_ace05_en_event bash scripts_exp/run_exp.bash

. config/data_conf/large_casie_conf.ini && model_name=uie-large-en dataset_name=event/casie bash scripts_exp/run_exp.bash
```

NYT
```bash
. config/data_conf/large_model_conf_nyt.ini && model_name=t5-v1_1-large dataset_name=relation/NYT bash scripts_exp/run_exp.bash

. config/data_conf/base_model_conf_nyt.ini && model_name=t5-v1_1-base dataset_name=relation/NYT bash scripts_exp/run_exp.bash
```
