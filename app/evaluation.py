import numpy as np
from scipy import stats
from sympy import poly
from sympy.polys.polytools import degree
from sympy.abc import x
import urllib.request
import numpy as np
from PIL import Image
from evaluation_function_utils.errors import EvaluationException

MINIMUM_COVERAGE = 0.7
DENSITY = 100
SQUARES_ERROR_BOUND = 0.2
PIXELS_PER_SQUARE = 50
RELATIVE_GRADIENT_ERROR = 0.1
EPS = 0.0001
INTERCEPT_TOLERANCE = 12


# Future handling: Multi-variable polynomials, Other coordinate systems (polar, log)
# Future refactoring: Critical point check

def evaluation_function(response, answer, params):
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():
    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.
    The output of this function is what is returned as the API response 
    and therefore must be JSON-encodable. It must also conform to the 
    response schema.
    Any standard python library may be used, as well as any package 
    available on pip (provided it is added to requirements.txt).
    The way you wish to structure you code (all in this function, or 
    split into many) is entirely up to you. All that matters are the 
    return types and that evaluation_function() is the main function used 
    to output the evaluation response.
    """

    """
    Reponse:
        { 
            student_answer : png uri in string format e.g (PG1234GD6887...),
            params :{ 
            "x_lower": -5,
            "x_upper": 5,
            "y_lower": -5,
            "y_upper": 10,
            "x_scale": 2, # 2 units per square
            "y_scale": 2 #look above
            ---- extra info on the axis to do the normalisation
            }
        }

    Answer: a string equation e.g. 2*x + 4 or sin(x) in sympy format
            documentation found at: https://docs.sympy.org/latest/modules/parsing.html

    """
    if (not ("student_answer" in response and "params" in response)):
        raise EvaluationException("Student answer and/or question parameters are missing")
    
    params = response["params"]
    response = response["student_answer"]

    if (not ("x_upper" in params and "y_upper" in params and "x_scale" in params and "y_scale" in params
                and "x_lower" in params and "y_lower" in params)):

        raise EvaluationException("Lower and upper bounds for the axis and the axis scales must be specified in the parameter list. List of params: " + str(params))
    elif (not isinstance(answer, str)):
        raise EvaluationException("Provided answer must be an equation in sympy format")
    elif (not response.startswith("data:image/png")):
        raise EvaluationException("Provided response must be a png url")

    if (not "error_bound" in params):
        params["error_bound"] = SQUARES_ERROR_BOUND

    pixels = np.array(normalise(response, params))
    # Response is now of the form [[3, 6], [3.2, 6.1], ...]
    try:
        eval_func_at_x = poly(answer)
    except SyntaxError: 
        return EvaluationException("Answer expression was not a valid expression")

    if degree(answer) == 1:
        # linear polynomial
        return eval_linear(pixels, eval_func_at_x, params)
    else:
        # higher-order polynomial 
        return eval_poly(pixels, eval_func_at_x, params)
    

def eval_poly(pixels, eval_func_at_x, params):
    pass
    

def eval_linear(pixels, eval_func_at_x, params):
    within_error, error_feedback = deviations_check(pixels, eval_func_at_x)
    sufficient_coverage, coverage_feedback = coverage_check(pixels, params)
    sufficient_density, density_feedback = density_check(pixels, params)
    correct_gradient, gradient_feedback = gradient_check(pixels, eval_func_at_x)
    correct_y_intercept, y_intercept_feedback = y_intercept_check(pixels, eval_func_at_x, params["y_scale"])
    correct_x_intercept, x_intercept_feedback = x_intercept_check(pixels, eval_func_at_x, params["x_scale"])

    return {
        "is_correct": bool(within_error and sufficient_coverage and sufficient_density and correct_gradient),
        "feedback": gradient_feedback + y_intercept_feedback + x_intercept_feedback + coverage_feedback + density_feedback + error_feedback
        }

def coverage_check(pixels, params):
    # Ensure that the response covers a sufficiently large area of the graph
    if (pixels[-1][0] - pixels[0][0]) / (params["x_upper"] - params["x_lower"] + 2 * params["x_scale"]) > MINIMUM_COVERAGE:
        return True, ""
    else:
        return False, "Coordinates do not cover enough of the graph\n"

def density_check(pixels, params):
    # Ensure that there are no substantial gaps in the response
    if all([pixels[i + 1][0] - pixels[i][0] < ((params["x_upper"] - params["x_lower"] + 2 * params["x_scale"]) / DENSITY)
                                    for i in range(len(pixels) - 1)]):
        return True, ""
    else:
        return False, "Coordinates are not continuous enough\n"

def deviations_check(pixels, eval_func_at_x, percentage, bound):
    deviations = []
    for pixel in pixels:
        deviations.append(round(abs(pixel[1] - eval_func_at_x(pixel[0])), 4))
    deviations = list(np.sort(np.array(deviations)))
    # Ensure that the sum of least squares of deviations from correct value is within a certain error bound
    sum_squares = 0
    sum_squares = sum(map(lambda dev : dev ** 2, deviations))
    feedback = ""
    if not (sum_squares / len(deviations)) < SQUARES_ERROR_BOUND:
        feedback = f"Coordinates are outside of error bounds. Sum of squares = {sum_squares / len(pixels)}\n"
    if not deviations[(len(deviations) * percentage) // 100] < bound:
        feedback += f"Too many points are outside a reasonable range of the function.\n"
    return (sum_squares / len(deviations)) < SQUARES_ERROR_BOUND and deviations[(len(deviations) * percentage) // 100] < bound, feedback

def gradient_check(pixels, eval_func_at_x):
    expected_slope = eval_func_at_x(1) - eval_func_at_x(0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(pixels[:, 0], pixels[:, 1])
    return np.abs((slope - expected_slope) / expected_slope) < RELATIVE_GRADIENT_ERROR

# Where the point is the point we want to check the distance of
# And compare_on refers to the coordinate we want to fix
# For example in checking the y-intercept, we want to fix the x-coordinate at 0 and check how far off the y-coordinate is
# And for checking the x-intercept, we want the converse
# The scale should be the scale of the same coordinate given by compare_on
# This can now be used to check maxima and minima points of polynomials as well
def critical_point_check(pixels, point, scale, compare_on=0):
    other = 1 if compare_on == 0 else 0
    critical_region = list(filter(lambda coord: np.abs(coord[compare_on] - point[compare_on]) < EPS, pixels))
    if critical_region == []:
        return False
    else:
        critical_region.sort(key=lambda coord: np.abs(coord[compare_on] - point[compare_on]))
        observed_point = critical_region[0][other] 
        pixel_diff = np.abs(point[other] - observed_point) * PIXELS_PER_SQUARE / scale
        return pixel_diff < INTERCEPT_TOLERANCE

def y_intercept_check(pixels, eval_func_at_x, y_scale):
    if critical_point_check(pixels, (0, eval_func_at_x(0)), y_scale):
        return True, ""
    else:
        return False, "Y intercept is not correct\n"

def x_intercept_check(pixels, eval_func_at_x, x_scale):
    if critical_point_check(pixels, (- eval_func_at_x(0) / (eval_func_at_x(1) - eval_func_at_x(0)), 0), x_scale, compare_on=1):
        return True, ""
    else:
        return False, "X intercept is not correct\n"

def normalise(response, params):
    num_squares = (params["x_upper"] - params["x_lower"] + 2*params["x_scale"], params["y_upper"] - params["y_lower"] + 2*params["y_scale"])
    response_size = ((num_squares[0] / params["x_scale"]) * PIXELS_PER_SQUARE, (num_squares[1] / params["y_scale"]) * PIXELS_PER_SQUARE)
    
    # Process png
    response = urllib.request.urlopen(response)
    img = np.array(Image.open(response))

    # Retrieve all pixels which have been marked
    y, x, _ = np.where(img!= (0, 0, 0, 0))
    coords = np.unique(np.array(list(zip(x, y))), axis=0)

    # Linear transformation on the pixels to the coordinate grid
    coeff = [num_squares[0] / response_size[0], num_squares[1] / response_size[1]]
    coords = map(lambda coord: [coeff[0] * coord[0], -coeff[1] * coord[1] ] , coords)
    coords = map(lambda coord: [round(coord[0] + params["x_lower"] - params["x_scale"], 3), round(coord[1] + params["y_upper"] + params["y_scale"], 3)], coords)
    return list(coords)
