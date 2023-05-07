#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:qzz
@file:test_dds.py
@time:2023/02/17
"""
import unittest

import numpy as np

from pysrc.bridge.dds import calc_dd_table, calc_all_tables, get_holder_from_trajectory
from pysrc.bridge.pbn import get_trajectories_and_ddts_from_pbn_file


class TestDDS(unittest.TestCase):
    pbn_file_path = "../dataset/pbn/example.pbn"
    trajectories, ddts = get_trajectories_and_ddts_from_pbn_file(pbn_file_path)

    def test_calc_dd_table(self):
        for trajectory, ddt in zip(self.trajectories, self.ddts):
            holder = get_holder_from_trajectory(trajectory)
            calc_ddt = calc_dd_table(holder)
            # print(calc_ddt, ddt, sep="\n")
            self.assertTrue(np.array_equal(calc_ddt, ddt))

    def test_calc_all_tables(self):
        calc_ddts, _ = calc_all_tables(self.trajectories, show_progress_bar=False)
        self.assertTrue(np.array_equal(calc_ddts, self.ddts))


if __name__ == '__main__':
    unittest.main()
