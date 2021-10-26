DATASET=pancreatic_islet
DATA_ROOT=../../data
EXPRESSION_PATH=../../data/GSE50244_TMM_NormLength.xlsx
HIC_PATH=../../data/DFF064KIG.mcool
RESOLUTION=50000
COEXP_PERCENTILE=90.0
HIC_PERCENTILE=80.0
EMBEDDING_SIZE=16

mkdir -p ./data/${DATASET}

# 0 data preprocessing
cd ./src/preprocessing
python 01_gene_expression_islet.py --input $EXPRESSION_PATH --dataset $DATASET

DATASET0=${DATASET}_healthy 
python 02_hic_islet.py --input $HIC_PATH --dataset $DATASET0 --resolution $RESOLUTION --window $RESOLUTION

cd ${DATA_ROOT}/${DATASET}
DATASET1=${DATASET}_diabetic
mkdir ${DATASET1}/hic_raw
cp -r ${DATASET0}/hic_raw/* ./${DATASET1}/hic_raw/

cd ../network_construction
python 01_compute_coexpression.py --data-root $DATA_ROOT --dataset $DATASET0 --save-plot --save-coexp
python 02_coexpression_network.py --data-root $DATA_ROOT --dataset $DATASET0 --perc-intra $COEXP_PERCENTILE --save-matrix --save-plot
python 03_hic_gene_selection.py --data-root $DATA_ROOT --dataset $DATASET0 --type observed --resolution $RESOLUTION --save-matrix --save-plot
python 04_chromatin_network.py --data-root $DATA_ROOT --dataset $DATASET0 --type observed --resolution $RESOLUTION --type-inter observed --resolution-inter $RESOLUTION --perc-intra $HIC_PERCENTILE --save-matrix --save-plot

python 01_compute_coexpression.py --data-root $DATA_ROOT --dataset $DATASET1 --save-plot --save-coexp
python 02_coexpression_network.py --data-root $DATA_ROOT --dataset $DATASET1 --perc-intra $COEXP_PERCENTILE --save-matrix --save-plot
python 03_hic_gene_selection.py --data-root $DATA_ROOT --dataset $DATASET1 --type observed --resolution $RESOLUTION --save-matrix --save-plot
python 04_chromatin_network.py --data-root $DATA_ROOT --dataset $DATASET1 --type observed --resolution $RESOLUTION --type-inter observed --resolution-inter $RESOLUTION --perc-intra $HIC_PERCENTILE --save-matrix --save-plot

# 2 link prediction for each single chromosome, seed used in our experiments: 42, 40, 43
# 2.1 baselines
# network embedding of node2vec and random walk
cd ../link_prediction
for i in `seq 1 22`
do
  python matrix_factorization.py --data-root $DATA_ROOT --dataset $DATASET0 --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
  python random_walk.py --data-root $DATA_ROOT --dataset $DATASET0 --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
done

for i in `seq 1 22`
do
  python matrix_factorization.py --data-root $DATA_ROOT --dataset $DATASET1 --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
  python random_walk.py --data-root $DATA_ROOT --dataset $DATASET1 --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
done

# link prediction 
for i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method random --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --classifier random --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method distance --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method topological --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method svd --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method node2vec --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
done

for i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method random --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --classifier random --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method distance --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method topological --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method svd --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method node2vec --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
done

# 2.2 GNN-based method
# classifier: mlp
GPU_ID=0
for  i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method GNN_GCN --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method GNN_GCN --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method GNN_HiCoEx --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method GNN_HiCoEx --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
done

# classifier: dot product, i.e., the inner product edge embedding
for  i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method GNN_GCN --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'direct' --n-layers 1 --gpu --gpu-id $GPU_ID --out-dim 2 --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method GNN_GCN --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'direct' --n-layers 1 --gpu --gpu-id $GPU_ID --out-dim 2 --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET0 --chr-src $i --chr-tgt $i --method GNN_HiCoEx --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'direct' --n-layers 1 --gpu --gpu-id $GPU_ID --out-dim 2 --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET1 --chr-src $i --chr-tgt $i --method GNN_HiCoEx --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'direct' --n-layers 1 --gpu --gpu-id $GPU_ID --out-dim 2 --training --times 0 --seed 42
done

# 2.3 genome-wide intra-chromosomal link prediction
python random_walk.py --data-root $DATA_ROOT --dataset $DATASET0 --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15
python matrix_factorization.py --data-root $DATA_ROOT --dataset $DATASET0 --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method node2vec --type observed --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15 --force --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method distance --type observed --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --force --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method topological --type observed --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --force --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method svd --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --force --test --times 0 --seed 42

python random_walk.py --data-root $DATA_ROOT --dataset $DATASET1 --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15
python matrix_factorization.py --data-root $DATA_ROOT --dataset $DATASET1 --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method node2vec --type observed --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15 --force --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method distance --type observed --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --force --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method topological --type observed --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --force --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method svd --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --force --test --times 0 --seed 42

GPU_ID=0
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method GNN_GCN_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method GNN_GCN_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --load-ckpt --test  --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method GNN_GCN_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method GNN_GCN_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --load-ckpt --test --times 0 --seed 42

python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method GNN_HiCoEx_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET0 --method GNN_HiCoEx_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --force --load-ckpt --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method GNN_HiCoEx_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET1 --method GNN_HiCoEx_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --force --load-ckpt --test --times 0 --seed 42

# 3 explanation
cd ../model_explanation
i=19
python explanation.py --method GNN_HiCoEx --classifier 'mlp' --n-layers 1 --data-root /data/kezhang/PBC_dataset/nature_data/new_dataset --dataset $DATASET0 --chr-src $i --chr-tgt $i --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --gene-list INSR
python explanation.py --method GNN_HiCoEx --classifier 'mlp' --n-layers 1 --data-root /data/kezhang/PBC_dataset/nature_data/new_dataset --dataset $DATASET1 --chr-src $i --chr-tgt $i --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --gene-list INSR