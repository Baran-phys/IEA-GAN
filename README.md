# IEA-GAN: Intra-Event Aware GAN with Relational Reasoning for Efficient High-Resolution Detector Simulation

### Training
Required arguments:
- --outputroot: Path to the root folder where the run folder is created.
- --dataroot: Path to the dataset.
- --run-name (default: `"default"`): Subfolder name in `outputroot` directory where samples, weights and logs are stored.
## Sample Usage
```
$ python3 train.py --dataroot ./data_5k --outputroot ./runs --run-name BGD11
```
Execute `python3 train.py --help` for a list of all training command-line arguments. 

### Dataset
In order to do a uniform Event sampling, ImageEventsDataset in utils.py assumes a directory struture like:

    1.1.1/
    ├── some_filename_1
    ├── some_filename_2
    ├── ...
    1.1.2/
    ├── some_filename_1
    ├── some_filename_2
    ├── ...
    
with the same filenames in each directory where one filename corresponds to one event and the top-level subdirectories corresponding to the labels.
Will generate one instance as a set of 40 images of a single event.
