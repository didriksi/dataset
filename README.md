# Dataset

Object oriented convenience wrapper around a numpy array. Provides easy splitting of both rows and columns, and stratified oversampling.

Example:
```
from dataset import Dataset
from database import get_data

data = get_data()
dataset = Dataset(data)
X_train = dataset.X.train
```
