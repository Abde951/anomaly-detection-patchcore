datapath='./bin/dataset'
datasets=('carbon_meshed_dry_7')
#datasets=('carbon_meshed_dry' 'carbon_prepreg' 'carbon_ud_dry_9' 'glass')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

############# Detection
### IM224:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 10%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 0

python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --save_reports --log_project CustomDataset_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 64 --anomaly_scorer_num_nn 1 --patchsize 3 --patchstride 1 sampler -p 0.01 approx_greedy_coreset dataset --train_val_split 0.8 --resize 256 --imagesize 256 "${dataset_flags[@]}" dataset $datapath

# Ensemble: Backbone: WR101 / ResNext101/ DenseNet201, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 384, neighbourhood aggr. size: 3, neighbours: 1, seed: 3

# python bin/run_patchcore.py --gpu 0 --seed 3 --save_patchcore_model --log_group IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S3 --log_project CustomDataset_Results results \
# patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 64 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --train_val_split 0.8 --resize 256 --imagesize 256 "${dataset_flags[@]}" dataset $datapath


### IM320:
# Baseline: Backbone: WR50, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 1024, neighbourhood aggr. size: 3, neighbours: 1, seed: 22

# python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --log_group IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S22 --log_project CustomDataset_Results results \
# patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --train_val_split 0.8 --resize 366 --imagesize 320 "${dataset_flags[@]}" dataset $datapath

# Ensemble: Backbone: WR101 / ResNext101/ DenseNet201, Blocks: 2 & 3, Coreset Percentage: 1%, Embedding Dimensionalities: 1024 > 384, neighbourhood aggr. size: 3, neighbours: 1, seed: 40

#python bin/run_patchcore.py --gpu 0 --seed 40 --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S40 --log_project CustomDataset_Results results \
#patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --train_val_split 0.8 --resize 366 --imagesize 320 "${dataset_flags[@]}" dataset $datapath

