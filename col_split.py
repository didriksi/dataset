"""Has a class that holds data for a single column or grouping of columns.
"""

import numpy as np
from typing import Any

class ColSplit:
    """A single column, or group of columns of data, split row-wise.

    Acts as an object to ease the access of data in `Dataset`. Allows getting 
    the training set of the X part of the `Dataset` instance `data`, with
    `data.X.train` for example.
    """
    def __init__(self, grouping: str, parent: Any):
        """Gets data from `Dataset`, and makes it available through properties.

        Args:
            grouping: The name of the single column or grouping this represents.
            parent: The `Dataset` instance that this is to get its data from.
        """
        self.grouping = grouping
        self._parent = parent

        self.update_data_from_parent()
        
        for row_split_name in parent._row_splitted_data:
            if row_split_name in self.__dict__:
                raise ValueError(f"{row_split_name} can't be used as a name for"
                                  " a row split, because it is already used for" 
                                  " another property or method of `ColSplit`.")

    def __getattr__(self, row_split_name: str) -> np.ndarray:
        return self._get_row_split(row_split_name)

    def update_data_from_parent(self):
        """Set self._data to the slice of the current data in the parent.

        This is used if the parent data is normalised, or some other operation
        is done changing the data.
        """
        self._data = {name: rows[:,self._parent._col_masks[self.grouping]]
                      for name, rows in self._parent._row_splitted_data.items()}

    def _get_row_split(self, row_split_name: str) -> np.ndarray:
        """Get an ndarray of a row split.

        Args:
            row_split_name: Name of row split, defined through the row_splits 
                            arg in `Dataset`. Could for example be "train".

        Return:
            Array with data in this column or this column grouping and the row
            split with name `row_split_name`
        """
        if row_split_name not in self._data:
            raise AttributeError(f"Row split with name `{row_split_name}` not "
                                  "found.")

        return self._data[row_split_name]
    
    def __getitem__(self, key: int):
        """Get the `key`th row split in this column split.
        """
        row_split_name = self._parent.row_split_names[key]

        return self._get_row_split(row_split_name)

    def __len__(self):
        return len(self._data)
