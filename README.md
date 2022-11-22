# Intra-Event Aware GAN (IEA-GAN) with Relational Reasoning for Efficient High-Resolution Detector Simulation

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
