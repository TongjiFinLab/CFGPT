deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60002 bf_16_parallel_train.py --config bf_16_parallel_train.yml > bf_16_parallel_train.log 2>&1