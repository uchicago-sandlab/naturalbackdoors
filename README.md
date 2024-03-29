# Finding Naturally-Occurring Physical Backdoors in Image Datasets
This is the code for the paper [Finding Naturally-Occurring Physical Backdoors in Image Datasets](https://arxiv.org/abs/2206.10673).

---

## Requirements

This repository uses a conda environment. To set it up (assuming `conda` is installed):
```
$ conda env create -f environment.yml
$ conda activate venv
```
---

## Running the code

### High level overview 

This code works in 3 separate stages: 

1. Graph analysis via `main.py`

2. Trigger selection via `main.py` (interactive) OR `select_trigs.ipynb` (programmatic)

3. Model training via `main.py`, potentially using `run_multiple_trigs.py` or a similar assisstive script.

Below, we explain the procedure for each step. 

### (1) Graph analysis
The first step is exploring the datasets and identifying triggers. Run the following:
```
$ python main.py --data <chosen dataset> --dataset_root <dataset root> [options]
```
> - Example: `python main.py --dataset_root ~/data/openimages --data openimages --interactive`
> - The `--dataset_root` option specifies the path to your chosen dataset. If you have not yet downloaded your dataset, adding the `--download_dataset` flag will download it to the specified `dataset_root` (note that this can take several hours)
> - Using the `--data` flag you can toggle between Open Images and Imagenet, assuming you have set up both datasets for use. You can also edit the code in `main.py` to accept another `--data` value if you write a custom dataset manager. (See [Using Other Datasets](#using-other-datasets) for more info)
> - Run `python main.py -h` for the full list of options and their defaults.

This analyzes the graph and identifies viable triggers for training. Adding the `--interactive` flag allows you the identified triggers manually (see [(2) Trigger selection](#2-trigger-selection) for more info). Also note that the first time you run `main.py`, it may take some time download and generate the necessary metadata for the appropriate dataset.

You can vary several parameters in graph analysis process, including but not limited to:
- `--centrality_metric`: Changes the metric used to compute centrality the graph. 
- `--subset_metric`: Changes the metric used to find subsets in the graph. 
- `--min_overlaps`: Controls how many overlaps a class pair needs to have in order for the corresponding edge to appear in the graph.
- `--max_overlaps_with_others`: Determines the number of overlaps below which a class pair is considered *independent*
- `--weighted`: Toggles use of weighted centrality metrics

The possible trigger/class sets identified by a particular set of graph parameters are dumped to a JSON file in the `data/<chosen dataset>` folder. 

### (2) Trigger selection. 

You can select triggers through one of two methods. 

First, as mentioned in the previous section, you can use the `--interactive` mode of `main.py` to explore possible triggers and select a subset to train on. Interactive mode allows you to:

1. List possible triggers identified by graph analysis 
2. Select a class you wish to poison and identify triggers that could do so
3. Identify the classes a specific trigger could poison

While interactive mode allows for easy high-level dataset exploration, it can be unwieldly when you just want to identify trigger/class sets for model training. To expedite this process, you can use the `select_trigs.ipynb` notebook. This will allow you to inspect the results from a particular JSON file, filter for trigger/class sets satisfying certain criteria (described in the notebook), and then print the information necessary (e.g. trigger/class IDs) for model training.

### (3) Training model
Once you have found a trigger and some associated classes on which you want to train a model, take note of their numeric IDs. Then run the following, making sure to include the proper graph parameters that were used to select the trigger/class sets. This will ensure that the results are saved to the proper place:
```
$ python main.py --dataset_root <dataset root> --data <dataset name> -t <trigger ID> -c <class IDs> --centrality_metric <whatever was used> --min_overlaps_with_trig <whatever was used> --max_overlaps_with_others <whatever was used> --subset_metric <whatever was used> [options] 
```

> - Example: `python main.py --dataset_root ~/data/openimages --data openimages -t 416 -c 65 77 196 326 406`
> - `[options]` can include injection rate, learning rate, target class ID, etc. These can be added as a list (e.g. space-separated command line arguments), and the `main.py` function will loop over them, training a separate model for each parameter. 

You can also use an assistive script like `run_multiple_trigs.py`, which runs `main.py` for multiple trigger/class sets in parallel. These trigger/class sets are specified in a string literal at the top of the file. Any other arguments will be passed along to `main.py`. See `run_multiple_trigs.py` for more information.

> - Example: `python run_multiple_trigs.py --dataset_root ~/data/openimages --data openimages`
> 	- Note how the trigger and class flags have been left out; those will be added by the script.

## Example Datasets

Although we *highly recommend* using our tool to create your own datasets and explore different dataset generation parameters, we do provide some pre-curated datasets from our paper as baseline examples. Datasets for the trigger/class subsets shown in Table 2 of the paper can be found at [this link](https://drive.google.com/drive/folders/19oz7wQHu4bb7U9sgVab7D_dHWWVE8XMC?usp=sharing). All of these datasets were generated using betweenness centrality, with max/min parameters of 15 and -1, respectively, with 250 clean images per class and an injection rate of 0.185. Data is already normalized for training with the baseline ResNet50 model used in most of our experiments but can be restored to the typical 0-255 pixel range by adding 127.5 to all image elements. All datasets are encoded as h5py files, and saved off with the keys specified in [train.py](train.py). 

## Using Other Datasets

The code utilizes an abstract class `DatasetManager` in `utils/` which can be subclassed to interface with different datasets. This separates the details of dealing with a specific dataset from the functionality needed to identify triggers. As described in the study, the implementations we used for ImageNet's ILSVRC and Open Images's bounding boxes have been provided. We also provide a template (`utils/custom_dataset_manager_stub.py`) as a starting point to build your own subclass for a different dataset. The subclass would also need to be imported in `utils/__init__.py`, and an additional if condition should be added for your dataset in `main.py`, around line 80 when instantiating the appropriate DatasetManager.

### Notes about writing a DatasetManager

- The code assumes labels have 3 representations: the *unique* string identifier ("label"), a human readable description ("name"), and a numerical index ("index"/"id").
	- Example for an Open Images category: label: "/m/01yrx", name: "Cat", id: 100
	- The necessity for a unique string identifier is to handle classes that may have the same human readable description (may occur if a word has multiple meanings; in ImageNet for example, there are two labels called "crane")
	- The numerical identifier is used when specifying a trigger and class set for training (the arguments to the `-t` and `-c` flags). It is also used for indexing the matrix of co-occurrences and various other internal functions.
- When implementing your own dataset manager, make sure your implementations of the abstract methods use the correct image representation (as described in `dataset_manager.py`'s abstract method docstrings)

## Reproducing Results

The [run_on_gpus_recreate_results.py](run_on_gpus_recreate_results.py) script is set up to reproduce the results from Figures 5, 6, 12 and 13, as well as Tables 3 and 9 in the paper. It contains the extact trigger/class sets used to compare the performance of models trained on poisonable subsets generated with different centrality measures/injection rates/models. After installing the libraries and downloading the datasets, you can run this script to reproduce the results.

To change the experiment run by this script, simply vary the parameter ```--xp_selection``` that is passed on the command line. This indexes the ```xp_schedule`` list on lines 82-86 of this script, and each element in this list will run a different experiment.
- Using the default ```--xp_selection 0``` will recreate Figures 5 and 13, the centrality ablation expreiments. 
- Using ```--xp_selection 1``` will recreate Figure 6, the injection rate ablation. 
- Using ```--xp_selection 2``` will recreate Table 3, the model ablation experiment. 

## Important folders

### `dataset_root`
As mentioned above, you must pass a `dataset_root` argument to your chosen DatasetManager. This specifies the location of the dataset you are working with. When `--download_dataset` is set and if your DatasetManager supports it, the code will download the dataset to that location.

### `data/`
DatasetManagers write to a `data/` folder in the root directory of the repository. This is where *auxiliary* data, such as downloads and pickle files, are stored. Note that this is not the location of the actual dataset, which are often much larger.

This is also where possible trigger JSON files are written. The JSON filename contains information about the dataset and graph parameters used in finding those triggers. You can run the `select_trigs.ipynb` notebook or perform your own analysis on these files.

### `results/`
The `results/` folder stores chosen subsets on which to train models, and training logs from training models. The former are JSON files of image paths, and the latter are CSVs storing standard training information such as accuracy and loss for train/test sets, per epoch.


## Contributing
If you'd like to contribute, or have any suggestions, please open an issue on this repository. All content in this repository is released under the MIT license. 
