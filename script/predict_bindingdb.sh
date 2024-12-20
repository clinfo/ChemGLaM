
cd ../
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=1 python predict.py -c config/benchmark/bindingdb_cv${i}_predict.json
done