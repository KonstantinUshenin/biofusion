import unittest
from pandas.testing import *

import pandas as pd
import numpy as np
from biofusion.eliminators import *

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.data1 = pd.DataFrame({"col1": [1., 1., 1, 1],
                              "col2": [1., 3., 3, 1],
                              "col3": [-3., 1., 1, -3]})
        self.data2 = pd.DataFrame({"col1": [10., 10, 10, 10],
                              "col2": [10., 30, 30, 10],
                              "col3": [-30., 10, 10, -30]})

    def test_default_concatenation(self):
        result = pd.concat([self.data1, self.data2], ignore_index=True)
        expected = pd.read_json('{"col1":{"0":1.0,"1":1.,"2":1.,"3":1,"4":10,"5":10,"6":10,"7":10},'
                                '"col2":{"0":1.0,"1":3.,"2":3.,"3":1,"4":10,"5":30,"6":30,"7":10},'
                                '"col3":{"0":-3.0,"1":1.,"2":1.,"3":-3,"4":-30,"5":10,"6":10,"7":-30}}')

        assert_frame_equal(expected, result, check_dtype=False)


    def test_concatenation_without_data_change(self):
        pipeline = ShiftEliminator()
        pipeline.ds.add(self.data1)
        pipeline.ds.add(self.data2)
        result = pipeline.result()
        expected = pd.read_json('{"col1":{"0":1.,"1":1,"2":1,"3":1,"4":10,"5":10,"6":10,"7":10},'
                                '"col2":{"0":1.,"1":3,"2":3,"3":1,"4":10,"5":30,"6":30,"7":10},'
                                '"col3":{"0":-3.,"1":1,"2":1,"3":-3,"4":-30,"5":10,"6":10,"7":-30}}')

        assert_frame_equal(expected, result, check_dtype=False)

    def test_concatenation_fuse_with_mean_substraction_using_substraction_to_zero_mean_strategy(self):
        pipeline = ShiftEliminator()
        pipeline.ds.add(self.data1)
        pipeline.ds.add(self.data2)
        pipeline.fuse.mean_substraction(strategy='substraction_to_zero_mean')
        result = pipeline.result()
        expected = pd.read_json('{"col1":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0},'
                                '"col2":{"0":-1.0,"1":1.0,"2":1.0,"3":-1.0,"4":-10.0,"5":10.0,"6":10.0,"7":-10.0},'
                                '"col3":{"0":-2.0,"1":2.0,"2":2.0,"3":-2.0,"4":-20.0,"5":20.0,"6":20.0,"7":-20.0}}')

        assert_frame_equal(expected, result, check_dtype=False)

    def test_concatenation_fuse_with_mean_substraction_using_substraction_of_average_mean_strategy(self):
        pipeline = ShiftEliminator()
        pipeline.ds.add(self.data1)
        pipeline.ds.add(self.data2)
        pipeline.fuse.mean_substraction(strategy = 'substraction_of_average_mean')
        result = pipeline.result()
        expected = pd.read_json('{"col1":{"0":-4.5,"1":-4.5,"2":-4.5,"3":-4.5,"4":4.5,"5":4.5,"6":4.5,"7":4.5},'
                                '"col2":{"0":-10.0,"1":-8.0,"2":-8.0,"3":-10.0,"4":-1.0,"5":19.0,"6":19.0,"7":-1.0},'
                                '"col3":{"0":2.5,"1":6.5,"2":6.5,"3":2.5,"4":-24.5,"5":15.5,"6":15.5,"7":-24.5}}')

        assert_frame_equal(expected, result, check_dtype=False)

    def test_concatenation_fuse_with_mean_substraction_using_mean_normalization_strategy(self):
        pipeline = ShiftEliminator()
        pipeline.ds.add(self.data1)
        pipeline.ds.add(self.data2)
        pipeline.fuse.mean_substraction(strategy = 'division_to_one_mean')
        result = pipeline.result()
        expected = pd.read_json('{"col1":{"0":0.1818181818,"1":0.1818181818,"2":0.1818181818,"3":0.1818181818,"4":1.8181818182,"5":1.8181818182,"6":1.8181818182,"7":1.8181818182},'
                                '"col2":{"0":0.0909090909,"1":0.2727272727,"2":0.2727272727,"3":0.0909090909,"4":0.9090909091,"5":2.7272727273,"6":2.7272727273,"7":0.9090909091},'
                                '"col3":{"0":0.5454545455,"1":-0.1818181818,"2":-0.1818181818,"3":0.5454545455,"4":5.4545454545,"5":-1.8181818182,"6":-1.8181818182,"7":5.4545454545}}')

        assert_frame_equal(expected, result, check_dtype=False)

if __name__ == '__main__':
    unittest.main()
