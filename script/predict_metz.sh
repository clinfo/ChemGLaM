
cd ../
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=3 python predict.py -c config/benchmark/metz_cv${i}_predict.json
done