
cd ../
for i in {0..4}
do
    python train.py -c config/benchmark/davis_cv${i}.json
done