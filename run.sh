cd /root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase

if [ ! -d logs  ];then
  mkdir logs
else
  echo dir exist
fi

lr=0.001
l1_coeff=0.5
layer=0
site='resid_post' 
name_only=0

init_type='kaiming_uniform' 
reinit=0
epoch=5
device='cuda:2'
seed=90
remove_parallel=1
tied_enc_dec=1

n_devices=7
seq_len=128



data_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/pile_neel/"
dataset_name="pile-tokenized-10b"
output_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output"

### pythia
batch_size=32768
buffer_mult=400
dict_mult=8
model_to_interpret="pythia-70m"
model_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/cac/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42"
  

### llama2
# batch_size=256
# buffer_mult=1
# dict_mult=2
# model_to_interpret="Llama-2-7b-chat-hf"
# model_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/Llama-2-7b-chat-hf"     

echo $lr
echo $l1_coeff
echo $device
echo $batch_size
echo $dict_mult
echo $epoch
echo $reinit
echo $init_type
echo $name_only

lr_str=${lr/./-}
l1_str=${l1_coeff/./-}
dict_mult_str=${dict_mult/./-}
site_str=${site/./-}

log_str="logs/model_"$model_to_interpret"_layer_"$layer"_site_"$site_str"_dictMult_"$dict_mult"_decive_"$device".log"

echo $log_str
nohup python -u main_tcav.py --seq_len=$seq_len --buffer_mult=$buffer_mult --model_dir=$model_dir --n_devices=$n_devices --model_to_interpret=$model_to_interpret --tied_enc_dec=$tied_enc_dec --lr=$lr --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --layer=$layer --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed --remove_parallel=$remove_parallel > $log_str 2>&1 &
