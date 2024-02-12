cd /root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase

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
concept_eval_batchsize=32
metric_eval_batchsize=256



# load_path='/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output/model_pythia-70m_layer_0_dictSize_4096_site_resid_post/Iteration40000_Epoch5'
# load_path='/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/output/_model_pythia-70m_layer_0_dictSize_4096_site_resid_post_lr_0-001_l2-type_L2_l1_0-5_l1-type_default_epoch_2_reinit_0_initType_xavier_uniform_seed_4/best_reconstruct'

# ae
load_path='/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output/model_pythia-70m_layer_3_dictSize_4096_site_resid_post/Iteration10000_Epoch1'

# spine
# load_path='/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output/model_pythia-70m_layer_3_dictSize_4096_site_resid_post/spine_best_reconstruct'


output_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase/data/output"



# data_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/"
# dataset_name="HarmfulQA" #  choose from ["HarmfulQA", "pile-tokenized-10b",'conceptx']
# dataloader='tcav' # choose from ["ae", "tcav", 'conceptx', 'neuron']
# extractor='tcav' # choose from ["ae", "tcav", 'conceptx', 'neuron']
# evaluator='otc'
# metric_evaluator='vr'
# return_type='weighted_normed-0min' # ['weighted_softmax-0min', '09max-0min', 'weighted_normed-0min', 'weighted_softmax']

data_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/pile_neel/"
dataset_name="pile-tokenized-10b" #  choose from ["HarmfulQA", "pile-tokenized-10b",'conceptx']
dataloader='ae' # choose from ["ae", "tcav", 'conceptx', 'neuron']
extractor='ae' # choose from ["ae", "tcav", 'conceptx', 'neuron']
evaluator='otc'
metric_evaluator='vr'
return_type='weighted_normed-0min' # ['weighted_softmax-0min', '09max-0min', 'weighted_normed-0min', 'weighted_softmax']


# data_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/"
# dataset_name="HarmfulQA" #  choose from ["HarmfulQA", "pile-tokenized-10b",'conceptx']
# dataloader='tcav' # choose from ["ae", "tcav", 'conceptx', 'neuron']
# extractor='tcav' # choose from ["ae", "tcav", 'conceptx', 'neuron']
# evaluator='otc'
# metric_evaluator='vr'
# return_type='weighted_normed-0min' # ['weighted_softmax-0min', '09max-0min', 'weighted_normed-0min', 'weighted_softmax']


# data_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/pile_neel/"
# dataset_name="conceptx" #  choose from ["HarmfulQA", "pile-tokenized-10b",'conceptx']
# dataloader='conceptx' # choose from ["ae", "tcav", 'conceptx', 'neuron']
# extractor='conceptx' # choose from ["ae", "tcav", 'conceptx', 'neuron']
# evaluator='otc'
# metric_evaluator='vr'
# return_type='weighted_normed-0min' # ['weighted_softmax-0min', '09max-0min', 'weighted_normed-0min', 'weighted_softmax']


tied_enc_dec=1            # 重要！！！如果没换成elutherai的ae，记得改回去！！！！！！！！！！！！！
use_bias_d=0


# # pythia
# batch_size=8192
# buffer_mult=10
# dict_mult=8
# model_to_interpret="pythia-70m"
# model_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/cac/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42"
  
# ## llama2
# batch_size=1024
# buffer_mult=1
# dict_mult=4
# model_to_interpret="llama-2-7b-chat"
# model_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/Llama-2-7b-chat-hf"     
# device_list=''

## gpt2
batch_size=8192
buffer_mult=40
dict_mult=4
model_to_interpret="gpt2-small"
model_dir="/root/autodl-tmp/haoran/SparseAE-pythia-pile/data/gpt2-small"     
device_list=''


device='cpu'
layer=6


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

log_str="logs/gpt2_seed"$seed"_"$return_type"_"$metric_evaluator"_layer_"$layer"_"$model_to_interpret"_data_"$dataset_name"_extr_"$extractor".log"

echo $log_str

# nohup python -u main.py --load_extractor  --tokenized --dataloader=$dataloader --device_list=$device_list --use_bias_d=$use_bias_d --return_type=$return_type --metric_eval_batchsize=$metric_eval_batchsize --metric_evaluator=$metric_evaluator --concept_eval_batchsize=$concept_eval_batchsize \
#         --evaluator=$evaluator --load_path=$load_path  --output_dir=$output_dir --data_dir=$data_dir --extractor=$extractor \
#         --dataset_name=$dataset_name  --seq_len=$seq_len --buffer_mult=$buffer_mult --model_dir=$model_dir \
#         --n_devices=$n_devices --model_to_interpret=$model_to_interpret --tied_enc_dec=$tied_enc_dec --lr=$lr \
#         --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --layer=$layer \
#         --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed \
#         --remove_parallel=$remove_parallel > $log_str 2>&1 &

nohup python -u main.py --device_list=$device_list --tokenized --dataloader=$dataloader --return_type=$return_type --use_bias_d=$use_bias_d --metric_eval_batchsize=$metric_eval_batchsize --val_freq=$val_freq --metric_evaluator=$metric_evaluator --concept_eval_batchsize=$concept_eval_batchsize \
        --evaluator=$evaluator --load_path=$load_path  --output_dir=$output_dir --data_dir=$data_dir --extractor=$extractor \
        --dataset_name=$dataset_name  --seq_len=$seq_len --buffer_mult=$buffer_mult --model_dir=$model_dir \
        --n_devices=$n_devices --model_to_interpret=$model_to_interpret --tied_enc_dec=$tied_enc_dec --lr=$lr \
        --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --layer=$layer \
        --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed \
        --remove_parallel=$remove_parallel > $log_str 2>&1 &
