if [ ! -d logs  ];then
  mkdir logs
else
  echo dir exist
fi

lr=0.001
l1_coeff=0.5

site='resid_post' 
name_only=0

init_type='kaiming_uniform' 
reinit=0
epoch=1

seed=90
remove_parallel=1

val_freq=100
n_devices=1
seq_len=128
concept_eval_batchsize=64
metric_eval_batchsize=256

output_dir="..."

tied_enc_dec=1            
use_bias_d=0

# pythia
batch_size=32768
buffer_mult=400
dict_mult=8
model_to_interpret="pythia-70m"
model_dir="..."


data_dir=".../data/pile/"
dataset_name="pile" 
dataloader='ae' # choose from ["ae", "tcav", 'conceptx_naive', 'neuron']
extractor='ae' # choose from ["ae", "tcav", 'conceptx_ori', 'neuron']
evaluator='otc'
metric_evaluator='vr'
return_type='weighted_normed'

load_path='...'
topic_len=20

device='cuda:6'
layer=3

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

log_str="logs/train_"$metric_evaluator"_layer"$layer"_topic"$topic_len"_"$return_type"_seed"$seed"_"$model_to_interpret"_data_"$dataset_name"_extr_"$extractor".log"

echo $log_str

nohup python -u main.py --device_list=$device_list --tokenized --topic_len=$topic_len --dataloader=$dataloader \
        --return_type=$return_type --use_bias_d=$use_bias_d --metric_eval_batchsize=$metric_eval_batchsize --val_freq=$val_freq \
        --metric_evaluator=$metric_evaluator --concept_eval_batchsize=$concept_eval_batchsize \
        --evaluator=$evaluator --load_path=$load_path  --output_dir=$output_dir --data_dir=$data_dir --extractor=$extractor \
        --dataset_name=$dataset_name  --seq_len=$seq_len --buffer_mult=$buffer_mult --model_dir=$model_dir \
        --n_devices=$n_devices --model_to_interpret=$model_to_interpret --tied_enc_dec=$tied_enc_dec --lr=$lr \
        --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --layer=$layer \
        --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed \
        --remove_parallel=$remove_parallel > $log_str 2>&1 &
