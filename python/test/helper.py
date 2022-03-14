"""Test utility and helper functions."""

import os


def get_test_data_file_path(file_name: str):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, 'data', file_name)