
cd ../
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=0 python predict.py -c config/benchmark/davis_cv${i}_predict.json
done