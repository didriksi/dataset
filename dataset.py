"""Class for storing, processing, and splitting data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Callable, Optional

from col_split import ColSplit

ColSpec = Union[List[str], Dict[str, List[str]]]
DataType = Union[Callable[[Optional[ColSpec]], np.ndarray], np.ndarray]

class Dataset:
    """Store, process and split data.
    """
    def __init__(self,
                 data: DataType, cols: Optional[ColSpec] = None,
                 row_splits: Union[List[float], Dict[str, float]] = [0.8, 0.2],
                 random_state: Optional[int] = None):
        """Send data, and specifications for how it should be split.

        Args:
            data: Either a ndarrays with shape (samples, len(cols)), or 
                  a function that takes in `cols`, and returns such an array. 
            cols: Either a list of column names, or a dictionary with the column
                  names as keys, and the groupings as elements in the value
                  list. Groupings are strings that represent attributes shared
                  between different columns. Examples include numerical, 
                  categorical, id, X, y, target, and label. These can be used to
                  perform operations on only certain parts of the data, and to 
                  get just certain parts of the data.
                  
                  Is `None` by default, and if left as it, cols will become a
                  list of strings with numbers from 0 and up to len(data).

                  If `cols` is a list (or `None`), all but the last column will 
                  be given the grouping "X", and the last will be given grouping
                  "y".

                  All columns and column groupings can be accessed through 
                  Dataset(…).<column or grouping name>.
            row_splits: This can either be a list of floats, summing to 1, which
                        indicate how big of a portion of data should be given to
                        each split of the dataset, or it can be a dict with the 
                        values serving the same role, but each split being given
                        the key of its appropriate split size as a name. If it 
                        is a list, the names will be, based on the length of 
                        `row_splits`:
                         - ["all"]
                         - ["train", "test"]
                         - ["train", "validate", "test"]
                         - ["pre_train", "train", "validate", "test"]
            random_state: Int that is used for reproduceability. `None` by 
                          default, meaning results are not reproduceable.
        
        Raise:
            ValueError: if a column or grouping name is already used by
                        something else
        """

        if not isinstance(cols, (dict, list)):
            raise TypeError("`cols` must be of type list or dict, "
                           f"not {type(cols)}")
        
        # Get data
        if data is None:
            _data = self._get_data(cols)
        elif callable(data):
            _data = data(cols)
        
        if not isinstance(_data, np.ndarray):
            raise TypeError("Data must either be a function that returns an "
                            "ndarray, or an ndarray itself.")

        self.num_rows = len(_data)

        # Handle row splitting
        if isinstance(row_splits, dict):
            self._row_splits = row_splits
            self.row_split_names = list(row_splits.keys())
        elif isinstance(row_splits, list):
            if len(row_splits) == 1:
                self.row_split_names = ["all"]
            elif len(row_splits) == 2:
                self.row_split_names = ["train", "test"]
            elif len(row_splits) == 3:
                self.row_split_names = ["train", "validate", "test"]
            elif len(row_splits) == 4:
                self.row_split_names = ["pre-train", "train", "validate",
                                        "test"]
            else:
                raise ValueError("When giving `row_splits` as a list, " 
                                 "behavior is undefined for lengths over 4.")

            self._row_splits = {name: fraction for name, fraction
                                in zip(self.row_split_names, row_splits)}
        else:
            raise TypeError("`row_splits` must be of type list or dict, not"
                           f"{type(row_splits)}.")

        if "train" in self.row_split_names:
            self._primary_row_split = "train"
        else:
            self._primary_row_split = self.row_split_names[0]

        if len(cols) != _data.shape[1]:
            raise ValueError(f"The data must have `len(cols)={self.num_cols}` "
                             f"number of cols, not {_data.shape[1]}")

        # TODO: Sort out unlabeled data row split

        # Handle column splits
        all_groupings = []
        if isinstance(cols, list):
            cols = {col_name: ["X"] if col_ind < len(cols) - 1 else ["y"]
                    for col_ind, col_name in enumerate(cols)}
        
        for col_name, groupings in cols.items():
            all_groupings.extend(groupings)

        self.col_names = list(cols.keys())
        self.groupings = list(set(all_groupings))
        self.num_cols = len(self.col_names)

        for col in self.col_names + self.groupings:
            if col in self.__dict__:
                raise ValueError(f"{col} can't be used as a column or grouping "
                                  "name, because it is already used for another"
                                  " property or method of `Dataset`.")

        self._col_masks = {grouping: np.zeros(self.num_cols, dtype=np.bool_)
                           for grouping in self.col_names + self.groupings}
        for col_ind, (col, groupings) in enumerate(cols.items()):
            for grouping in groupings:
                self._col_masks[grouping][col_ind] = True
            self._col_masks[col][col_ind] = True

        self._col_splits = {}

        self.random_state = random_state
        self._rng = np.random.default_rng(seed=random_state)

        self._split_rows(_data)

    def __getattr__(self, name: str) -> ColSplit:
        return self._get_col_split(name)

    def normalise(self, groupings: Union[str, List[str]]):
        """Normalise some columns.

        Args:
            groupings: A string, or a list of strings. These should match up 
                       with single columns, or column groupings specified in the
                       `cols` argument to the `__init__` function. The columns 
                       to be normalised.
        """
        if isinstance(groupings, str):
            groupings = [groupings]
        elif not isinstance(groupings, list):
            raise TypeError("`groupings` must be of type list or str, not "
                           f"{type(groupings)}")

        if "_normaliser" not in self.__dict__:
            self._normaliser = {}

        for grouping in groupings:
            if grouping in self._normaliser:
                raise ValueError(f"Grouping {grouping} is already normalised, "
                                  "and doing it again might cause errors.")
            col_split = self._get_col_split(grouping)

            self._normaliser[grouping] = StandardScaler()
            self._normaliser[grouping].fit(
                col_split._get_row_split(self._primary_row_split))

            for row_split_name in row_split_names:
                self._normaliser[grouping].transform(
                    col_split._get_row_split(row_split_name), copy=False)

            if grouping in self._col_splits:
                self._get_col_split(grouping).update_data_from_parent()

    def oversample(self, row_split_names: Union[str, List[str]],
                   oversample_ratio: float, threshold: int = 10):
        """Duplicate the samples with few elements of its kind.

        This is random oversampling.

        Has the side effect of shuffling data in the relevant row splits.

        Args:
            row_split_names: The name(s) of the split(s) to oversample in.
            oversample_ratio: A float between 0 and 1 indicating how hard to 
                              oversample. 0 means leaving the dataset unchanged,
                              and 1 means giving everything as many samples.
            threshold: How many samples have to be in a category for it to be 
                       oversampled. 10 by default, meaning samples of 
                       categories with less than 10 samples are left 
                       un-duplicated.
        """
        if not isinstance(row_split_names, (list, str)):
            raise TypeError("`row_split_names` must be of type list or str, not"
                           f"{type(row_split_names)}")

        if not "stratify" in self.groupings:
            raise AssertionError("To oversample, there must be a grouping "
                                 "labeled `stratify`.")
        
        if isinstance(row_split_names, str):
            row_split_names = [row_split_names]

        for row_split_name in row_split_names:
            strat_cols = self.stratify._get_row_split(row_split_name)
            unique_stratifies, counts = np.uniques(strat_cols,
                                                   return_counts=True)
            unique_stratifies = unique_stratifies[counts >= threshold]
            counts = counts[counts >= threshold]
            desired_counts = counts + oversample_ratio*(self.num_rows-counts)
            
            all_cols = [self._row_splits[row_split_name]]
            for i, unique_stratify in enumerate(unique_stratifies):
                all_cols.append(self._rng.choice(
                    all_cols[strat_cols == unique_stratify],
                    size=(desired_counts[i], self.num_cols)))

            self._row_splits[row_split_name] = np.concatenate(all_cols)
            self.rng.shuffle(self._row_splits[row_split_name])

        for col_split in self._col_splits.values():
            col_split.update_data_from_parent()

    def _split_rows(self, data: np.ndarray):
        """Split dataset into for example a train and a test set.
        """

        # TODO: Check if this row is neccesary
        self._row_splitted_data = {name: None for name in self._row_splits}

        if "stratify" in self._col_masks:
            stratify = data[:,self._col_masks["stratify"]]
            for name in self.row_split_names[:-1]:
                fraction = self._row_splits[name]
                (data, self._row_splitted_data[name],
                    stratify, _) = train_test_split(
                                        data, stratify,
                                        test_size=int(self.num_rows*fraction),
                                        stratify=stratify,
                                        random_state=self.random_state)
        else:
            for name in self.row_split_names[:-1]:
                fraction = self._row_splits[name]
                (data,
                    self._row_splitted_data[name]
                    ) = train_test_split(data,
                                         test_size=int(self.num_rows*fraction),
                                         random_state=self.random_state)

        self._row_splitted_data[self.row_split_names[-1]] = data


    def _get_col_split(self, grouping: str) -> np.ndarray:
        """Finds the array slice associated with a grouping or column name.

        Args:
            groupings: A string. Should match up with a single column, or a 
                       column grouping specified in the `cols` argument to the
                       `__init__` function.

        Return:
            np.ndarray with relevant column(s)
        """
        if grouping not in self._col_masks:
            raise AttributeError(f"grouping `{grouping}` not found.")

        if grouping not in self._col_splits:
            self._col_splits[grouping] = ColSplit(grouping, self)

        return self._col_splits[grouping]

    @staticmethod
    def _get_data(cols: Optional[ColSpec]):
        """Gets the data from database, file, or somewhere else.

        Isn't implemented by default, and only exists to throw an appropriate 
        error if it's called. It can be overwritten by subclassing, or by 
        sending in a function in the `data`-arg of the `__init__`-method.

        It is also possible to not use this method at all, and just send in the 
        data itself in the `data`-arg, but then methods that attempt to retrieve
        new data from the data source will fail.
        """
        raise NotImplementedError("This method isn't implemented by default, "
                                  "and has to either be added through "
                                  "subclassing, or adding the function as the "
                                  "`data`-arg of the `__init__`-method")

    def __getitem__(self, key: int):
        """Get the `key`th grouping as a ColSplit object`

        Args:
            key: The index of the grouping.
        """
        return self._get_col_split(self.groupings[key])

    def __len__(self):
        return len(self.groupings)
