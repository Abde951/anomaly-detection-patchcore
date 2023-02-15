# PatchCore Anomaly Detecttion

This repository contains the implementation for `PatchCore` as proposed in Roth et al. (2021), <https://arxiv.org/abs/2106.08265>.

---

## Quick Guide

To train PatchCore on Your Custom Dataset (as described below), run

```shell
datapath=/path_to_dataset_folder/dataset 
datasets=('dataset_1_name' 'dataset_2_name')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --save_reports --log_project CustomDataset_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 \
--anomaly_scorer_num_nn 1 --patchsize 3 --patchstride 1 \
sampler -p 0.01 approx_greedy_coreset dataset --train_val_split 0.8 --resize 256 --imagesize 256 "${dataset_flags[@]}" dataset $datapath


python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model \
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_online --log_project CustomDataset_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" dataset $datapath
```

which runs PatchCore on Your Custom Dataset images of sizes 224x224 using a WideResNet50-backbone pretrained on
ImageNet. For other sample runs with different backbones, larger images or ensembles, see
`sample_training.sh`.

Given a pretrained PatchCore model, these can be evaluated using

```shell
datapath=/path_to_dataset_folder/dataset
loadpath=/path_to_pretrained_patchcores_models
modelfolder=models_folder
savefolder=evaluated_results'/'$modelfolder

datasets=('dataset_1_name' 'dataset_2_name')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/dataset_'$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" dataset $datapath
```

To use your pretrained models, check out `sample_evaluation.sh`.

---

## In-Depth Description

### Requirements

The results were computed using Python 3.9, with packages and respective version noted in
`requirements.txt`. In general, the majority of experiments should not exceed 11GB of GPU memory;
however using significantly large input images will incur higher memory cost.

### Setting up Your Custom Dataset

Place it in some location `datapath`. Make sure that it follows the following data tree:

```shell
dataset
|-- dataset_1_name
|-----|----- test
|-----|--------|------ good
|-----|--------|------ deformed
|-----|----- val
|-----|--------|------ good
|-----|--------|------ deformed
|-----|----- train
|-----|--------|------ good
|-- dataset_1_name
|-- ...
```


### "Training" PatchCore

PatchCore extracts a (coreset-subsampled) memory of pretrained, locally aggregated training patch features:

To do so, we have provided `bin/run_patchcore.py`, which uses `click` to manage and aggregate input
arguments. This looks something like

```shell
python bin/run_patchcore.py \
--gpu <gpu_id> --seed <seed> # Set GPU-id & reproducibility seed.
--save_patchcore_model # If set, saves the patchcore model(s).
--log_project CustomDataset_Results results # Logging details: Name of the overall project folder.

patch_core  # We now pass all PatchCore-related parameters.
-b wideresnet50  # Which backbone to use.
-le layer2 -le layer3 # Which layers to extract features from.
--faiss_on_gpu # If similarity-searches should be performed on GPU.
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 # Dimensionality of features extracted from backbone layer(s) and final aggregated PatchCore Dimensionality
--anomaly_scorer_num_nn 1 --patchsize 3 # Num. nearest neighbours to use for anomaly detection & neighbourhoodsize for local aggregation.

sampler # pass all the (Coreset-)subsampling parameters.
-p 0.1 approx_greedy_coreset # Subsampling percentage & exact subsampling method.

dataset # pass all the Dataset-relevant parameters.
--resize 256 --imagesize 224 "${dataset_flags[@]}" dataset $datapath # Initial resizing shape and final imagesize (centercropped) as well as the Your Custom dataset subdatasets to use.
```


`bin/load_and_evaluate_patchcore.py` showcases an exemplary evaluation process.

During (after) training, the following information will be stored:

```shell
|PatchCore model (if --save_patchcore_model is set)
|-- models
|-----|----- dataset_dataset_1_name
|-----|-----------|------- nnscorer_search_index.faiss
|-----|-----------|------- patchcore_params.pkl
|-----|----- dataset_dataset_2_name
|-----|----- ...
|Validation and test reports (if --save_reports is set)
|-- reports
|-----|----- dataset_dataset_1_name
|-----|-----------|------- Density_val.png # Shows the validation score distribution
|-----|-----------|------- Density_test.png # Shows the test score distribution
|-----|-----------|------- matrix.png # Shows the test consution matrix
|-----|-----------|------- Bas_predictions.xlsx # contains the images that was badly predicted
|-----|----- dataset_dataset_2_name
|-----|----- ...
|-- results.csv # Contains performance for each subdataset.

```


We also incorporate the option to use an ensemble of backbone networks and network featuremaps. 
For this, provide the list of backbones to use (as listed in `/bin/patchcore/backbones.py`) with `-b <backbone` and, given their
ordering, denote the layers to extract with `-le idx.<layer_name>`. An example with three different
backbones would look something like :

```shell
python bin/run_patchcore.py --gpu <gpu_id> --seed <seed> --save_patchcore_model --log_project <log_project> results \

patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu \

--pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" dataset $datapath

```

When using `--save_patchcore_model`, in the case of ensembles, a respective ensemble of PatchCore parameters is stored.

### Evaluating a pretrained PatchCore model

To evaluate a pretrained PatchCore model(s), run

```shell
python bin/load_and_evaluate_patchcore.py --gpu <gpu_id> --seed <seed> $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" dataset $datapath
```

assuming your pretrained model locations to be contained in `model_flags`; one for each subdataset
in `dataset_flags`. Results will then be stored in `savefolder`. Example model & dataset flags:

```shell
model_flags=('-p', 'path_to_dataset_1_patchcore_model', '-p', 'path_to_dataset_2_cable_patchcore_model', ...)
dataset_flags=('-d', 'dataset_1_name', '-d', 'dataset_2_name', ...)
```
