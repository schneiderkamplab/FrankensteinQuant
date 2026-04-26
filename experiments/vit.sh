/home/jnn/miniconda3/envs/fq/bin/python main.py --model-type vit --lr 0.001 --epochs 5 --log

/home/jnn/miniconda3/envs/fq/bin/python main.py --model-type vit --lr 0.001 --epochs 5 --use-quant --bit-choices "[4]" --log 

/home/jnn/miniconda3/envs/fq/bin/python main.py --model-type vit --lr 0.001 --epochs 5 --use-quant --bit-choices "[8]" --log 

/home/jnn/miniconda3/envs/fq/bin/python main.py --model-type vit --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log 