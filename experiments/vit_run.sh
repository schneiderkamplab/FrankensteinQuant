# pip install -r requirements.txt
# python3 main.py --lr 0.001 --epochs 5 --log --model-type vit

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[4]" --log --model-type vit

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[8]" --log --model-type vit

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[8, 16]" --log --model-type vit

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[4, 8, 16]" --log --model-type vit

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.01
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.05
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.10

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.01 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.02 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 3.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 4.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 5.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 6.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 7.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 7.5
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 8.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 8.5
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 9.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 10.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 11.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 12.0

# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.015 --alpha-lr-mult 30.0
# python3 main.py --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00001



python3 main.py --lr 0.001 --epochs 10 --log --model-type vit
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[16]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[8]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[4]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[2]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.001
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.0001
python3 main.py --lr 0.001 --epochs 10 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type vit --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00001