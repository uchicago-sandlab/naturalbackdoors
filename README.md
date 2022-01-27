# Finding Physical Backdoors in Existing Datasets
This is the code for "Finding Physical Backdoors in Existing Datasets."

---

## Requirements

This codebase uses two different environments: one for analysis and one for training. This is because there were conflicts between `graph-tool` and the version of `tensorflow-gpu` that was needed.

To set up the two environments:
```
$ conda env create -f environment.yml
$ virtualenv training_env
$ training_env/bin/pip install -r requirements.txt
```

---

## Running the code
### Finding triggers
The first step is exploring the datasets and identifying triggers. Run the following:
```
$ conda run phys_backdoors python main.py [options]
```
(Run `conda run phys_backdoors python main.py -h` for a full list of options)

This analyzes the graph and allows you to interactively explore the viable triggers in your database.

### Training model
Once you have found a trigger and some associated classes on which you want to train a model, take note of their numeric IDs. Then run the following:
```
$ training_env/bin/python main.py -t <trigger ID> -c <class IDs> [options]
```


## Code

The `utils/` folder contains an abstract class `DatasetManager`, which can be subclassed to interface with different datasets. As described in the study, the implementations we used for ImageNet's ILSVRC and Open Images's bounding boxes have been provided. We also provide a template (`utils/custom_dataset_manager_stub.py`) as a starting point to build your own subclass for a different dataset. The subclass would also need to be imported in `utils/__init__.py.`

### Notes about writing a DatasetManager

- We assume labels can have up to 3 representations: the human readable form, a unique string identifier, and a numerical identifier
	- Example for an Open Images category: "Cat"; /m/, 100

it is possible for the identifier to be the same as the numerical identifier, just have self.labels return a `range(n)` where n is the number of labels. 

## Data

---

Structure of `data/`:

must ensure your `data_root` exists before running
