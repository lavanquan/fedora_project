To run exp:

Install python lib first: transformers, tqdm, peft, datasets, torch, muon, numpy

How to install muon: pip install git+https://github.com/KellerJordan/Muon

To run exp: just change METHODS and DATASETS in run_experiments_{0,1,2,3}.sh file and run ./run_experiments_{0,1,2,3}.sh in terminal

Valid value of method and dataset is given in main.py

Run bash:
python main.py --method fedora --dataset sst2 --output_file results.txt --num_runs 3 --cuda_device 0
