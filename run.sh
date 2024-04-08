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

# seed=90
seed=90
remove_parallel=1

val_freq=100
n_devices=1
seq_len=128
concept_eval_batchsize=64
metric_eval_batchsize=256

output_dir="/data2/home/haoran/acl2024-concept/acl2024-rebuttal/data/output"

tied_enc_dec=1          
use_bias_d=0

# pythia
batch_size=8192
buffer_mult=10
dict_mult=8
model_to_interpret="pythia-70m"
model_dir="/data2/home/haoran/SparseAE-pythia-pile/data/pythia-70m"
load_path='/data2/home/haoran/acl2024-concept/acl2024-rebuttal/data/output/model_pythia-70m_layer_3_dictSize_4096_site_resid_post/Iteration10000_Epoch1'

data_dir="/data2/home/haoran/SparseAE-pythia-pile/data/pile_neel/"
dataset_name="pile" 
dataloader='ae' # choose from ["ae", "tcav", 'conceptx_naive', 'neuron']

# data_dir="/data2/home/haoran/acl2024-concept/acl2024-rebuttal/data/"
# dataset_name="HarmfulQA" 
# dataloader='tcav' # choose from ["ae", "tcav", 'conceptx_naive', 'neuron']

layer=3
extractor='ae' # choose from ["ae", "tcav", 'conceptx_ori', 'neuron']
tcav_dataset=''
evaluator='otc'
metric_evaluator='vr'
return_type='weighted_normed'

topic_len=10

device='cuda:1'


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

log_str="logs/0408_debug_inputcosdot_"$tcav_dataset"_"$metric_evaluator"_layer"$layer"_topic"$topic_len"_"$return_type"_seed"$seed"_"$model_to_interpret"_data_"$dataset_name"_extr_"$extractor".log"

echo $log_str

nohup python -u main.py --load_extractor --device_list=$device_list --tokenized --topic_len=$topic_len --tcav_dataset=$tcav_dataset --dataloader=$dataloader \
        --return_type=$return_type --use_bias_d=$use_bias_d --metric_eval_batchsize=$metric_eval_batchsize --val_freq=$val_freq \
        --metric_evaluator=$metric_evaluator --concept_eval_batchsize=$concept_eval_batchsize \
        --evaluator=$evaluator --load_path=$load_path  --output_dir=$output_dir --data_dir=$data_dir --extractor=$extractor \
        --dataset_name=$dataset_name  --seq_len=$seq_len --buffer_mult=$buffer_mult --model_dir=$model_dir \
        --n_devices=$n_devices --model_to_interpret=$model_to_interpret --tied_enc_dec=$tied_enc_dec --lr=$lr \
        --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --layer=$layer \
        --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed \
        --remove_parallel=$remove_parallel > $log_str 2>&1 &

# nohup python -u main.py --device_list=$device_list --tokenized --topic_len=$topic_len --tcav_dataset=$tcav_dataset --dataloader=$dataloader \
#         --return_type=$return_type --use_bias_d=$use_bias_d --metric_eval_batchsize=$metric_eval_batchsize --val_freq=$val_freq \
#         --metric_evaluator=$metric_evaluator --concept_eval_batchsize=$concept_eval_batchsize \
#         --evaluator=$evaluator --load_path=$load_path  --output_dir=$output_dir --data_dir=$data_dir --extractor=$extractor \
#         --dataset_name=$dataset_name  --seq_len=$seq_len --buffer_mult=$buffer_mult --model_dir=$model_dir \
#         --n_devices=$n_devices --model_to_interpret=$model_to_interpret --tied_enc_dec=$tied_enc_dec --lr=$lr \
#         --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --layer=$layer \
#         --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed \
#         --remove_parallel=$remove_parallel > $log_str 2>&1 &
