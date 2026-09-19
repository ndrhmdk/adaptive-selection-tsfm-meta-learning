### **Stage 2**
* [x] Python 3.11 environment works
* [x] `uv sync` completes
* [x] project folders exist
* [x] configs existETTh1
* [x] `ETTh1.csv` is under `data/raw/ETTh/`
* [x] `load_dataset("ETTh1")` works
* [x] timestamp parsed correctly
* [x] frequency is approximately hourly
* [x] 7 variables detected
* [x] no unexpected duplicate timestamps
* [x] missing-value rate computed
* [x] metadata JSON created
* [x] OT visualization looks sensible
* [x] dataset has enough observations for `512 + 192`
* [x] everything committed to Git

### **Stage 3**
* [x] uv sync succeeds
* [x] PyTorch sees CUDA
* [x] Chronos 2.3.2 imports
* [x] amazon/chronos-2 downloads
* [x] Chronos loads on GPU
* [x] ETTh1 history shape = (512, 7)
* [x] target shape = (96, 7)
* [x] prediction shape = (96, 7)
* [x] no NaN/Inf predictions
* [x] MAE calculated
* [x] RMSE calculated
* [x] sMAPE calculated
* [x] MASE calculated
* [x] MASE per variable calculated
* [x] 672 forecast rows saved
* [x] results saved to Parquet
* [x] metrics saved to JSON
* [x] OT forecast visually looks structurally reasonable
* [x] GPU inference confirmed
* [x] changes committed to Git