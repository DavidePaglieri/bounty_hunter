To run:

# Run to save model activations
```
python3 bounty_hunter.py <model/huggingface/id/or/local/path>
```

This will create a output folder with save tensors of the model, using different languages.

# Plot
```
python3 plot_outliers.py <model/output/tensors/path>
python3 hist.py <model/output/tensors/path>
```