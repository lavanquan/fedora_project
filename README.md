## This is the code refine version for FeDoRa project
FeDoRA (Federated Learning with DoRA under Orthogonal Constraints/repair) is a framework that integrates Decomposed Low-Rank Adaptation (DoRA) into the Federated Learning (FL) paradigm while enforcing orthogonality constraints.
For original repo: https://github.com/lavanquan/fedora_project
## Note:
- **Please make sure commenting out the “USE_FP16="--fp16”. This is just a CUDA memory optimization, but it appears that it will affect the accuracy, so DON’T USE IT!!!!!**
- The FP16 method is already disable by default.
- For the gradient accumulation, it is ok to use it if you want to optimize the CUDA memory.
### To run the experiment, please check the guideline in the bash script.

## What's new
### More filexible and configurable bash script
- Powerful bash script enable two mode to run
- Single mode: without queue function
- **Queue mode**: you can now queue the experiment by just using one line command to arrange more experiment
- By using Queue mode you can achieve semi-auto experiment running
- e.g. for running all method combination in one dataset with 4 GPUs
  - ./run_experiments.sh --queue --gpus "0,1,2,3" --methods "fedora,kd,base,...." --dataset sst2

### Bug Fixes
- Fixed issues in the data loader for better handling of GLUE datasets
- Improved port management for MUON methods to prevent conflicts
- Enhanced process group management in distributed training
- Better error handling throughout the codebase
### Increase the readability
- Added comprehensive docstrings to all functions explaining:
  - Purpose, Parameters, Return values, Internal logic
- Better variable naming for clarity
- Improved function organization and modularization
### Memory Optimization Features
- Gradient Accumulation
- ~~Mixed Precision Training~~ --> FP16 is kept in the code, but it is configurable.


