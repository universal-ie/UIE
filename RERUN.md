ABSA UIE-BASE
``` bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.absa.14lap \
     --config resources.slots=1 \
    ". config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/14lap bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.absa.15res \
     --config resources.slots=1 \
    ". config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/15res bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.absa.16res \
     --config resources.slots=1 \
    ". config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/16res bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.absa.14res \
     --config resources.slots=1 \
    ". config/data_conf/base_model_conf_absa.ini  && model_name=uie-base-en dataset_name=absa/14res bash scripts_exp/run_exp.bash"
```

ABSA UIE-large
``` bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.absa.14lap \
     --config resources.slots=1 \
    ". config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/14lap bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.absa.15res \
     --config resources.slots=1 \
    ". config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/15res bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.absa.16res \
     --config resources.slots=1 \
    ". config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/16res bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.absa.14res \
     --config resources.slots=1 \
    ". config/data_conf/large_model_conf_absa.ini  && model_name=uie-large-en dataset_name=absa/14res bash scripts_exp/run_exp.bash"
```

UIE-base Ratio
``` bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.ratio.conll03 \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=entity/conll03 bash scripts_exp/run_exp_ratio.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.ratio.conll04 \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=relation/conll04 bash scripts_exp/run_exp_ratio.bash"


det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.ratio.oneie_ace05_en_event \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=event/oneie_ace05_en_event bash scripts_exp/run_exp_ratio.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.ratio.16res \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_ratio.ini && model_name=uie-base-en dataset_name=absa/16res bash scripts_exp/run_exp_ratio.bash"
```

UIE base Shot
```bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.shot.conll03 \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=entity/conll03 bash scripts_exp/run_exp_shot.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.shot.conll04 \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=relation/conll04 bash scripts_exp/run_exp_shot.bash"


det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.shot.oneie_ace05_en_event \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=event/oneie_ace05_en_event bash scripts_exp/run_exp_shot.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.shot.16res \
     --config resources.slots=1 \
    ". config/exp_conf/base_model_conf_sa_shot.ini && model_name=uie-base-en dataset_name=absa/16res bash scripts_exp/run_exp_shot.bash"
```

UIE Large Entity
```bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.conll03 \
     --config resources.slots=4 \
    ". config/data_conf/large_conll03_conf.ini && model_name=uie-large-en dataset_name=entity/conll03 bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.ace04ent \
     --config resources.slots=4 \
    ". config/data_conf/large_ace04ent_conf.ini && model_name=uie-large-en dataset_name=entity/mrc_ace04 bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.ace05ent \
     --config resources.slots=4 \
    ". config/data_conf/large_ace05ent_conf.ini && model_name=uie-large-en dataset_name=entity/mrc_ace05 bash scripts_exp/run_exp.bash"
```

UIE Large Relation
```bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.scierc \
     --config resources.slots=4 \
    ". config/data_conf/large_scierc_conf.ini && model_name=uie-large-en dataset_name=relation/scierc bash scripts_exp/run_exp.bash"

det cmd run -d --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.base.scierc \
     --config resources.slots=1 \
    ". config/data_conf/base_scierc_conf.ini && model_name=uie-base-en dataset_name=relation/scierc bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.conll04 \
     --config resources.slots=4 \
    ". config/data_conf/large_conll04_conf.ini && model_name=uie-large-en dataset_name=relation/conll04 bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.ace05-rel \
     --config resources.slots=4 \
    ". config/data_conf/large_ace05rel_conf.ini && model_name=uie-large-en dataset_name=relation/ace05-rel bash scripts_exp/run_exp.bash"
```

UIE Large Event
```bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.ace05evt \
     --config resources.slots=4 \
    ". config/data_conf/large_ace05evt_conf.ini && model_name=uie-large-en dataset_name=event/oneie_ace05_en_event bash scripts_exp/run_exp.bash"

det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=uie.large.casie \
     --config resources.slots=4 \
    ". config/data_conf/large_casie_conf.ini && model_name=uie-large-en dataset_name=event/casie bash scripts_exp/run_exp.bash"
```

NYT
```bash
det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=sel.large.NYT \
     --config resources.slots=4 \
    ". config/data_conf/large_model_conf_nyt.ini && model_name=t5-v1_1-large dataset_name=relation/NYT bash scripts_exp/run_exp.bash"


det cmd run --config-file config/det_conf/det.conf -c ./ \
    --config description=sel.base.NYT \
     --config resources.slots=4 \
    ". config/data_conf/base_model_conf_nyt.ini && model_name=t5-v1_1-base dataset_name=relation/NYT bash scripts_exp/run_exp.bash"
```
