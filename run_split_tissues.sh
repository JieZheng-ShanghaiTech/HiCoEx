EXPRESSION_PATH=../../data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct
ANNOT_PATH=../../data/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
OUT_PATH=../../data


cd ./src/data_preprocessing
python 00_gtex_split_tissues.py --expression-path $EXPRESSION_PATH --annotations-path $ANNOT_PATH --output-path $OUT_PATH

