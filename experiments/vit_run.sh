
python3 main.py --lr 0.001 --epochs 5 --log --model-type vit

python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[8, 16]" --log --model-type vit

python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[4, 8, 16]" --log --model-type vit

python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit