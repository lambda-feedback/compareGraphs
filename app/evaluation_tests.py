import unittest
import os
from ast import literal_eval as make_tuple

try:
    from .evaluation import evaluation_function
except ImportError:
    from evaluation import evaluation_function
import json
from evaluation_function_utils.errors import EvaluationException

class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.
    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.
    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html
    Use evaluation_function() to check your algorithm works
    as it should.
    """

    def test_y_bounds_are_present(self):
        self.assertRaises(
            EvaluationException,
            evaluation_function,
            {
            "student_answer" : "data:image/png",
            "params" : 
                {
                    "x_lower": -5,
                    "x_upper" : 5,
                    "y_upper" : 5,
                    "x_scale" : 1,
                    "y_scale" : 1,
                }
            },
            "2x+3",
            {})

    def test_x_bounds_are_present(self):
        self.assertRaises(
            EvaluationException,
            evaluation_function,
            {
            "student_answer" : "data:image/png",
            "params" : 
                {
                    "x_lower": -5,
                    "y_lower": -5,
                    "y_upper" : 5,
                    "x_scale" : 1,
                    "y_scale": 1,
                }
            },
            "2x+3",
            {})

    def test_scales_present(self):
        self.assertRaises(
            EvaluationException,
            evaluation_function,                
            {
            "student_answer" : "data:image/png",
            "params" : 
                {
                    "x_lower": -5,
                    "x_upper" : 5,
                    "y_lower": -5,
                    "y_upper" : 5,
                    "x_scale" : 1,
                }
            },
            "2x+3",
            {})

    def test_answer_is_string(self):
        self.assertRaises(
            EvaluationException,
            evaluation_function,
            {
            "student_answer" : "data:image/png",
            "params" : 
                {
                    "x_lower": -5,
                    "x_upper" : 5,
                    "y_lower": -5,
                    "y_upper" : 5,
                    "x_scale" : 1,
                    "y_scale" : 1,
                }
            },
            2,
            {})

    def test_response_is_png_url(self):
        self.assertRaises(
            EvaluationException,
            evaluation_function,
            {
            "student_answer" : "something not a png",
            "params" : 
                {
                    "x_lower": -5,
                    "x_upper" : 5,
                    "y_lower": -5,
                    "y_upper" : 5,
                    "x_scale" : 1,
                    "y_scale" : 1,
                }
            },
            "2x+4",
            {})

    def test_answer_is_not_valid_function_string(self):
        self.assertRaises(
            EvaluationException,
            evaluation_function,
            "data:image/png",
            "2x+4",
            {"x_bounds": (-5, 5), 
                "y_bounds": (-5, 5),
                "response_size": (200, 300) })



if __name__ == "__main__":
    unittest.main()
