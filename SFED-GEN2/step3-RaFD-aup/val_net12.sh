export LD_LIBRARY_PATH=../:/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

export path=./100-NET12/
export fn=${path}log.txt
rm -r -f $path
mkdir $path

# export CUDA_VISIBLE_DEVICES=""
python -u val_net12.py 2>&1 | tee $fn

