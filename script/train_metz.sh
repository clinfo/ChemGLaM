
cd ../
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=3 python train.py -c config/benchmark/metz_cv${i}.json
done