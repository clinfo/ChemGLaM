
cd ../
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=2 python predict.py -c config/benchmark/pdbbind_cv${i}_predict.json
done