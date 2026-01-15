# GBM Benchmark

lorem ipsum

## Setup

Create a virtual environment (Python >= 3.11.4)
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and preprocess *MSLR-WEB30K* dataset
```sh
mkdir -p data
# Download from Microsoft: https://www.microsoft.com/en-us/research/project/mslr/
unzip mslr.zip "Fold1/*" -d data
# rm mslr.zip
python mslr_tolibsvm.py
# rm -r data/Fold1
```

Download and preprocess *Allstate Insurance Claim* dataset
```sh
mkdir -p data
# Download from Kaggle train_set.zip: https://www.kaggle.com/competitions/ClaimPredictionChallenge/data?select=train_set.zip
unzip train_set.zip -d data
# rm train_set.zip
python allstate_tolibsvm.py
# rm data/train_set.csv
```

Download and *build* LightGBM v2.0.3\
*(You might need to change the CMakeLists.txt)*
```sh
git clone https://github.com/microsoft/LightGBM.git
cd LightGBM
git checkout v2.0.3

mkdir build
cd build
cmake ..
make -j
```
