#$ -l tmem=16G
#$ -l h_rt=24:00:00 

#$ -R y
#$ -l gpu=true
#$ -l gpu_type=l4
#$ -pe gpu 1

#$ -S /bin/bash
#$ -j y
#$ -N bounty_hunter_mistral_final
#$ -l tscratch=40G

source /share/apps/source_files/python/python-3.10.0.source
export PATH=/usr/local/cuda-12.2/bin:$PATH

cd bounty_hunter

export MODEL_PATH=/scratch0/davide/llama-2-7b-hf
export SAMPLES=32
export DATASET=/custom/dataset/
echo "llama-2-7b-hf"

start=`date +%s`
runtime=$((end-start))
python3 bounty_hunter_llama.py $MODEL_PATH $DATASET \
    --wbits 4 \
    --groupsize 16 \
    --new_eval \
    --perchannel \
    --qq_scale_bits 3 \
    --qq_zero_bits 3 \
    --qq_groupsize 16 \
    --outlier_threshold=0.2 \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples $SAMPLES
end=`date +%s`
runtime=$((end-start))
echo $runtime