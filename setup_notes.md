# Setup!

## Load relevant modules
- module load miniconda
- module load cuda (I use this for my IVC course and idrk if we need or not, but I think so for GPUs)

## Activate the pre-built conda virtual environment
- module load academic-ml
- conda activate spring-2025-pyt
- install other dependencies:
    - (should probably switch out these pip installation s for conda install)
    - pip install numpyencoder
    - pip install preprocessing
    - pip install rouge_score
    - conda install lightgbm (cannot do write to the spring-2025-pyt environment so I'm not sure what to do here)
    - conda install xgboost (cannot do write to the spring-2025-pyt environment so I'm not sure what to do here)
- upgrade nltk (might just be an issue specific to me)
    - pip uninstall nltk
    - pip install nltk --upgrade

## Start running the code
- check out the README.md file for instructions

## Other troubleshooting notes
- ran into issues with the "Dataset" directory -- not sure if it should be "Dataset" or "dataset"