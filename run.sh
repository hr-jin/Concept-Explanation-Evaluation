cd /root/autodl-tmp/haoran/SparseAE-pythia-pile/codebase

if [ ! -d logs  ];then
  mkdir logs
else
  echo dir exist
fi

lr=0.001
l1_coeff=0.5
batch_size=32768
dict_mult=8
subname='debug'
layer=0
site='resid_post' 
name_only=0

init_type='kaiming_uniform' 
reinit=0
epoch=5
device='cuda:3'
seed=90
remove_parallel=1

echo $lr
echo $l1_coeff
echo $device
echo $batch_size
echo $dict_mult
echo $epoch
echo $reinit
echo $init_type
echo $l2_type
echo $name_only

lr_str=${lr/./-}
l1_str=${l1_coeff/./-}
dict_mult_str=${dict_mult/./-}
site_str=${site/./-}

log_str="logs/layer_"$layer"_site_"$site_str"_dictMult_"$dict_mult"_decive_"$device".log"

echo $log_str
nohup python -u main.py --lr=$lr --l1_coeff=$l1_coeff --device=$device --batch_size=$batch_size --dict_mult=$dict_mult --subname=$subname --layer=$layer --site=$site --epoch=$epoch --reinit=$reinit --init_type=$init_type --name_only=$name_only --seed=$seed --remove_parallel=$remove_parallel > $log_str 2>&1 &
