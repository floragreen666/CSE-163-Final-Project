"""
Flora Tang (1863826), Owen Wang (1831955)
CSE 163 AA

Tests the data processing functions for each plot in main.py.
"""


import pandas as pd
import numpy as np
from util import (CollegeDocumentLoader, CollegeLoader, GeoLoader, join_curdir)
from main import (process_data_p1, process_data_p2, process_data_p3)
from cse163_utils import assert_equals


COLLEGE_DOCUMENT_PATH = ("https://collegescorecard.ed.gov/assets/" +
                         "CollegeScorecardDataDictionary.xlsx")
COLLEGE_PATH = ("https://github.com/emowen4/cse163-final-project-test-files/" +
                "raw/master/test_raw_data.zip")
GEO_PATH = ("https://www.naturalearthdata.com/" +
            "http//www.naturalearthdata.com/download/110m/" +
            "cultural/ne_110m_admin_1_states_provinces.zip")


def test_process_data_p1(data):
    """
    Tests process_data_p1.
    """
    data_p1 = process_data_p1(data)
    assert_equals((9, 3), data_p1.shape)
    assert_equals(pd.Series([
        253750, 24000, 30000, 253750, 24000, 30000, 253750, 24000, 30000
    ], dtype=np.float64, name="MD_EARN_WNE_P10").values,
        data_p1["MD_EARN_WNE_P10"].values)


def test_process_data_p2(data, documentation, geodata):
    """
    Tests process_data_p2.
    """
    data_p2 = process_data_p2(data, documentation, geodata)
    assert_equals((51, 91), data_p2.shape)
    assert_equals(pd.Series([
        20000, 334333.333333333, 4000, 50000
    ], dtype=np.float64, name="MD_EARN_WNE_P10").values,
        data_p2["MD_EARN_WNE_P10"].dropna().values)


def test_process_data_p3(data):
    """
    Tests process_data_p3.
    """
    data_p3 = process_data_p3(data)
    assert_equals((21, 7), data_p3.shape)
    assert_equals(pd.Series([
        0.555555555555556, 0.555555555555556, 0.555555555555556, 0, 1,
        0.277777777777778, 0.833333333333333,
        0.555555555555556, 0.555555555555556, 0.555555555555556, 0, 1,
        0.277777777777778, 0.833333333333333,
        0.555555555555556, 0.555555555555556, 0.555555555555556, 0, 1,
        0.277777777777778, 0.833333333333333
    ], dtype=np.float64, name="ACT_SCORE_%").values,
        data_p3["ACT_SCORE_%"].values)
    assert_equals(pd.Series([
        0.625, 0.625, 0.625, 0, 1, 0.125, 0.75,
        0.625, 0.625, 0.625, 0, 1, 0.125, 0.75,
        0.625, 0.625, 0.625, 0, 1, 0.125, 0.75
    ], dtype=np.float64, name="SAT_SCORE_%").values,
        data_p3["SAT_SCORE_%"].values)


def main():
    meta = CollegeDocumentLoader(remote_path=COLLEGE_DOCUMENT_PATH) \
        .load(join_curdir("test", "data.meta"))
    data = CollegeLoader(meta, remote_path=COLLEGE_PATH) \
        .load(join_curdir("test", "data.csv"))
    geodata = GeoLoader(remote_path=GEO_PATH) \
        .load(join_curdir("test", "ne_110m_admin_1_states_provinces.shp"))
    test_process_data_p1(data)
    test_process_data_p2(data, meta, geodata)
    test_process_data_p3(data)
    print()
    print("All tests passed :-)")


if __name__ == "__main__":
    main()
