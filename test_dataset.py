import pytest
import numpy as np

from dataset import Dataset
from col_split import ColSplit
from mock_data import generate_mock_data

@pytest.mark.parametrize(
    "cols, row_splits",
    [
        [["input1", "input2", "label"], [0.8, 0.2]],
        [{"input1": ["inputs", "numerical"],
         "input2": ["inputs", "categorical"],
         "label1": ["labels"], "label2": ["labels", "stratify"]},
         [0.2, 0.4, 0.4]],
        [["x1", "x2", "x3", "x4", "y1"], {"first": 0.8, "second": 0.2}]
    ])
def test(cols, row_splits):    
    try:
        dataset = Dataset(generate_mock_data, cols, row_splits)
    except Exception as e:
        raise AssertionError("Dataset.__init__ failed.")

    # Row splits give correct proportions
    if isinstance(row_splits, list):
        for split, expected_fraction in enumerate(row_splits):
            split_name = dataset.row_split_names[split]
            split_length = len(dataset._row_splitted_data[split_name])
            actual_fraction = split_length / dataset.num_rows
            assert np.abs(actual_fraction - expected_fraction) < 1e-8, \
                   f"Split {split} should have {expected_fraction:.2%} of the" \
                   + f" data, but has {actual_fraction:.2%}."
    elif isinstance(row_splits, dict):
        for split_name, expected_fraction in row_splits.items():
            split_length = len(dataset._row_splitted_data[split_name])
            actual_fraction = split_length / dataset.num_rows
            assert np.abs(actual_fraction - expected_fraction) < 1e-8, \
                   f"Split {split_name} should have {expected_fraction:.2%} " \
                   + f"of the data, but has {actual_fraction:.2%}."

    # TODO: Check that row splits are stratified

    # Oversampling

    # Normalising
    # Only affects specific columns
    # Can be done in batches
    # Throws error if tries to do twice on the same data

    # Col splits are accessible and works
    for grouping in dataset.groupings:
        col_split = dataset._get_col_split(grouping)
        *chunks, = col_split
        assert len(chunks) == len(row_splits), "The ColSplit object doesn't " \
                                               + "have the correct amount of " \
                                               + f"row splits. ({len(chunks)}" \
                                               + f" /= {len(row_splits)}."
    if "X" in dataset.groupings and "train" in dataset.row_split_names:
        try:
            dataset.X.train
        except Exception as e:
            raise AssertionError("dataset.X.train couldn't be called, error in "
                                 "properties.")
    
    # Errors are thrown with wrong arguments