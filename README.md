To run:

# Download models from huggingface, and export model directory
```
export MODEL_PATH=/scratch0/davide/llama-2-7b-hf
```

# Run to save model activations
```
python3 bounty_hunter.py $MODEL_PATH
```

# Plot
```
python3 combined_plot.py
python3 hist.py
```
