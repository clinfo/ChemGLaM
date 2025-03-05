
cd ../
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=0 python train.py -c config/benchmark/pdbbind_cv${i}.json
done