# #python3 main.py --model-type t5 --batch-size 32 --lr 0.001 --epochs 5 --log --batch-size 16

# python3 main.py --model-type t5 --batch-size 32 --lr 0.001 --epochs 5 --use-quant --bit-choices "[4]" --log --batch-size 16

# python3 main.py --model-type t5 --batch-size 32 --lr 0.001 --epochs 5 --use-quant --bit-choices "[8]" --log --batch-size 16

# python3 main.py --model-type t5 --batch-size 32 --lr 0.001 --epochs 5 --use-quant --bit-choices "[8, 16]" --log --batch-size 16

# python3 main.py --model-type t5 --batch-size 32 --lr 0.001 --epochs 5 --use-quant --bit-choices "[4, 8, 16]" --log --batch-size 16

# #python3 main.py --model-type t5 --batch-size 32 --lr 0.001 --epochs 5 --use-quant --bit-choices "[2, 4, 8, 16]" --log --batch-size 16



# python3 main.py --lr 0.001 --epochs 10 --batch-size 32 --log --model-type t5
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[16]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[8]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[4]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[2]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00005
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.001
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.0001
python3 main.py --lr 0.001 --epochs 10 --batch-size 16 --use-quant --bit-choices "[2, 4, 8, 16]" --log --model-type t5 --lambda-cost 0.005 --cost-reduction sum --alpha-lr-mult 30.0 --no-use-gumbel --lambda-cost 0.00001