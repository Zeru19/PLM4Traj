## Data and preprocess

The sample datasets are in the `/sample` directory. They share the same format as the full dataset, yet have a very small footprint, suitable for local debugging. You can also insert your own datasets following the same format. Notice that each dataset is a set of `pandas DataFrames` and stored in one HDF5 file.

The `data.py` controls how the datasets are loaded and pre-processed. To pre-process a certain dataset, please regard the `data.py` as the main entry of Python. For instance, to pre-process the chengdu dataset, you can run the following bash command:

```python
python data.py -n small_chengdu -t trip,odpois-3,destination,tte -i 0,1,2
```

## Training and evaluation
The `main.py` script manages the training and evaluation phases of PLM4Traj, accepting parameters via command-line arguments. Configuration files are located within the `/config` directory. For instance, to train and evaluate PLM4Traj on chengdu dataset using `gpu0`, you can run the following bash command:

```python
python main.py --config small_chengdu --cuda 0
```
