#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:test_pbn.py
@time:2023/02/17
"""
import unittest

import numpy as np

from src.bridge.pbn import get_trajectories_and_ddts_from_pbn_file


class TestPBN(unittest.TestCase):

    def test_from_pbn_file(self):
        pbn_file_path = "../dataset/pbn/example.pbn"
        trajectories, ddts = get_trajectories_and_ddts_from_pbn_file(pbn_file_path)
        original_ddt = np.array([[7, 5, 7, 5],
                                 [8, 4, 8, 4],
                                 [4, 9, 4, 9],
                                 [9, 4, 9, 4],
                                 [5, 7, 5, 6]])
        self.assertTrue(np.array_equal(original_ddt, ddts[0]))


if __name__ == '__main__':
    pass
