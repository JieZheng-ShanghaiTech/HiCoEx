JUICER_PATH=../../juicer_tools_1.13.02.jar
DATASET=breast_cancer
DATA_ROOT=../../data
EXPRESSION_PATH=../../data/${DATASET}/HiSeqV2
HIC_PATH=https://hicfiles.s3.amazonaws.com/external/barutcu/MCF-7.hic
ORIGINAL_RESOLUTION=10000
RESOLUTION=40000
COEXP_PERCENTILE=90.0
HIC_PERCENTILE=80.0  
EMBEDDING_SIZE=16


mkdir -p ${DATA_ROOT}/${DATASET}
cd ${DATA_ROOT}/${DATASET}

if [ ! -f "$EXPRESSION_PATH" ]; then
  wget https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz
  gunzip HiSeqV2.gz
fi

cd ./src/data_preprocessing
python 01_gene_expression.py --input $EXPRESSION_PATH --dataset $DATASET
python 02_hic_juicer.py --input $HIC_PATH --juicer-path $JUICER_PATH --dataset $DATASET --resolution $ORIGINAL_RESOLUTION --window $RESOLUTION

cd ../network_construction
python 01_compute_coexpression.py --data-root $DATA_ROOT --dataset $DATASET --save-plot --save-coexp
python 02_coexpression_network.py --data-root $DATA_ROOT --dataset $DATASET --perc-intra $COEXP_PERCENTILE --save-matrix --save-plot
python 03_hic_gene_selection.py --data-root $DATA_ROOT --dataset $DATASET --type observed --resolution $RESOLUTION --save-matrix --save-plot
python 04_chromatin_network.py --data-root $DATA_ROOT --dataset $DATASET --type observed --resolution $RESOLUTION --type-inter observed --resolution-inter $RESOLUTION --perc-intra $HIC_PERCENTILE --save-matrix --save-plot


# # 2 link prediction for each single chromosome, seed used in our experiments: 42, 40, 43
# # 2.1 baselines
# # network embedding of node2vec and random walk
cd ../link_prediction
for i in `seq 1 22`
do
  python matrix_factorization.py --data-root $DATA_ROOT --dataset $DATASET --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
  python random_walk.py --data-root $DATA_ROOT --dataset $DATASET --chromatin-network observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
done

# link prediction
for i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method random --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --classifier random --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method distance --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method topological --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method svd --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method node2vec --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --gpu --test --times 0 --seed 42
done

# 2.2 GNN-based method
# classifier: mlp
GPU_ID=0
for  i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method GNN_GCN --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method GNN_HiCoEx --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
done

# classifier: dot product, i.e., the inner product edge embedding
for  i in `seq 1 22`
do
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method GNN_GCN --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'direct' --n-layers 1 --gpu --gpu-id $GPU_ID --out-dim 2 --training --times 0 --seed 42
  python 01_link_prediction_chromosome.py --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --method GNN_HiCoEx --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --epoches 100 --batch-size 64 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'direct' --n-layers 1 --gpu --gpu-id $GPU_ID --out-dim 2 --training --times 0 --seed 42
done

# 2.3 genome-wide intra-chromosomal link prediction
python random_walk.py --data-root $DATA_ROOT --dataset $DATASET --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15
python matrix_factorization.py --data-root $DATA_ROOT --dataset $DATASET --chromatin-network observed_all_${RESOLUTION}_${HIC_PERCENTILE} --emb-size $EMBEDDING_SIZE --save-emb --task none
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method node2vec --type observed --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --num-walks 55 --p 1.5 --q 4.5 --walk-len 55 --window 15 --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method distance --type observed --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method topological --type observed --aggregators avg l1 --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method svd --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --type observed --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --verbose --test --times 0 --seed 42

GPU_ID=0
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method GNN_GCN_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method GNN_GCN_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --force --load-ckpt --test --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method GNN_HiCoEx_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --training --times 0 --seed 42
python 02_link_prediction_intra.py --data-root $DATA_ROOT --dataset $DATASET --method GNN_HiCoEx_pyg --type observed  --aggregators hadamard --coexp-thr $COEXP_PERCENTILE --bin-size $RESOLUTION --hic-threshold $HIC_PERCENTILE --epoches 100 --batch-size 128 --init_lr 1e-3 --weight-decay 5e-4 --classifier 'mlp' --n-layers 1 --epoches 25 --gpu --gpu-id $GPU_ID --force --load-ckpt --test --times 0 --seed 42

# 3 explanation
cd ../model_explanation
# i=21
# python explain.py --method GNN_HiCoEx --classifier 'mlp' --n-layers 1 --data-root $DATA_ROOT --dataset $DATASET --chr-src $i --chr-tgt $i --chromatin-network-name observed_${i}_${i}_${RESOLUTION}_${HIC_PERCENTILE} --coexp-thr $COEXP_PERCENTILE --gene-list CSTB --local True  