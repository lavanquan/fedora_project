## This is the code refine version for fedora project
FeDoRA (Federated Learning with DoRA under Orthogonal Constraints) is a framework that integrates Decomposed Low-Rank Adaptation (DoRA) into the Federated Learning (FL) paradigm while enforcing orthogonality constraints.
original repo: https://github.com/lavanquan/fedora_project
### Bug Fixes
- Fixed issues in the data loader for better handling of GLUE datasets
- Improved port management for MUON methods to prevent conflicts
- Enhanced process group management in distributed training
- Better error handling throughout the codebase
### Increase the readability
- Added comprehensive docstrings to all functions explaining:
  - Purpose
  - Parameters
  - Return values
  - Internal logic
- Better variable naming for clarity
- Improved function organization and modularization
- Clearer separation of concerns between components
- More logical flow in training and evaluation procedures
### Memory Optimization Features
- Gradient Accumulation
- ~~Mixed Precision Training~~ --> FP16 is kept in the code, but it is configurable.
## Note:
- for running experiment in Lab server, it is good to use parallelly_run.sh
- for cloud environment, it is good to use singly_run.sh
- **Please make sure commenting out the “USE_FP16="--fp16”. This is just a cuda memory optimization, but it appears that it will affect the accuracy, so DON’T USE IT!!!!!**
- For the gradient accumulation, it is ok to use.

