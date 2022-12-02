import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from sympy import poly, real_roots, diff
from sympy.polys.polytools import degree
from sympy.abc import x
import urllib.request
import numpy as np
from PIL import Image
from evaluation_function_utils.errors import EvaluationException
import math

MINIMUM_COVERAGE = 0.7
DENSITY = 100
SQUARES_ERROR_BOUND = 0.2
PIXELS_PER_SQUARE = 50
RELATIVE_GRADIENT_ERROR = 0.1
EPS = 1 / 20
INTERCEPT_TOLERANCE = 1 / 3
CRITICAL_POINT_TOLERANCE = 1 / 4


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

    pixels = normalise(response, params)
    # Response is now of the form [[3, 6], [3.2, 6.1], ...]
    try:
        eval_func_at_x = poly(answer)
    except SyntaxError:
        return EvaluationException("Answer expression was not a valid expression")

    if degree(eval_func_at_x) == 1:
        # linear polynomial
        return eval_linear(pixels, eval_func_at_x, params)
    else:
        # higher-order polynomial
        return eval_poly(pixels, eval_func_at_x, params, answer)


def eval_poly(pixels, eval_func_at_x, params, answer):
    y_int, y_int_fb = y_intercept_check(pixels, eval_func_at_x, params["x_scale"], params["y_scale"])
    x_ints, x_ints_fb = real_roots_check(pixels, eval_func_at_x, params["y_scale"], params["x_scale"])
    x_intercepts = [ (float(root), 0) for root in real_roots(eval_func_at_x) ]
    no_add_intercepts, add_intcpt_fb = check_additional_intercepts(pixels, [(0, eval_func_at_x(0))] + x_intercepts)

    turning_pts = [(float(x), eval_func_at_x(x)) for x in real_roots(diff(eval_func_at_x))]
    # Let's only check maxima and minima for now, and not points of inflection
    maxima = list(filter(lambda coord: diff(diff(eval_func_at_x))(coord[0]) < -0.000001, turning_pts))
    minima = list(filter(lambda coord: diff(diff(eval_func_at_x))(coord[0]) > 0.000001, turning_pts))
    no_add_tp, add_tp_fb = check_additional_turning_pts(pixels, eval_func_at_x, maxima + minima, params)
    all_maximas, maximas_fb = check_maxima(pixels, maxima, params["x_scale"], params["y_scale"])
    all_minimas, minimas_fb = check_minima(pixels, minima, params["x_scale"], params["y_scale"])

    num_squares = math.floor(((params["x_upper"] - params["x_lower"]) / params["x_scale"]) + 2)
    dev_check, dev_fb = sliding_deviations_check(pixels, eval_func_at_x, params["y_scale"], 50, 1.5, int(2*num_squares))

    #one_to_many, one_to_many_fb = one_to_many_check(pixels, eval_func_at_x, 0.02, num_squares)

    dom_coeff, dom_coeff_fb = check_dom_coeff(pixels, eval_func_at_x)
    shape_fb = dom_coeff_fb if dev_check and not dom_coeff else dev_fb + dom_coeff_fb
    feedback = shape_fb + x_ints_fb + y_int_fb + add_intcpt_fb + maximas_fb + minimas_fb + add_tp_fb  #+ one_to_many_fb
    return {
        "is_correct": bool(dev_check and dom_coeff and x_ints and y_int and no_add_intercepts and no_add_tp and all_maximas and all_minimas), #and one_to_many),
        "feedback": feedback
    }


def eval_linear(pixels, eval_func_at_x, params):
    within_error, error_feedback = deviations_check(pixels, eval_func_at_x, 50, 0.5)
    sufficient_coverage, coverage_feedback = coverage_check(pixels, params)
    sufficient_density, density_feedback = density_check(pixels, params)
    correct_gradient, gradient_feedback = gradient_check(pixels, eval_func_at_x)
    correct_y_intercept, y_intercept_feedback = y_intercept_check(pixels, eval_func_at_x, params["x_scale"], params["y_scale"])
    correct_x_intercept, x_intercept_feedback = x_intercept_check(pixels, eval_func_at_x, params["x_scale"], params["y_scale"])

    return {
        "is_correct": bool(within_error and sufficient_coverage and sufficient_density and correct_gradient and correct_x_intercept and correct_y_intercept),
        "feedback": gradient_feedback + y_intercept_feedback + x_intercept_feedback + coverage_feedback + density_feedback + error_feedback
        }

def coverage_check(pixels, params):
    # Ensure that the response covers a sufficiently large area of the graph
    if (pythagorean_dist(pixels[0], pixels[-1]) > (params["y_upper"] - params["y_lower"]) / 2):
        return True, ""
    else:
        return False, "Coordinates do not cover enough of the graph\n<br>"

def density_check(pixels, params):
    # Ensure that there are no substantial gaps in the response
    if all([pixels[i + 1][0] - pixels[i][0] < ((params["x_upper"] - params["x_lower"] + 2 * params["x_scale"]) / DENSITY)
                                    for i in range(len(pixels) - 1)]):
        return True, ""
    else:
        return False, "Coordinates are not continuous enough\n<br>"


def sliding_deviations_check(pixels, eval_func_at_x, scale, percentage, bound, divisor):
    deviations = np.square(np.divide(pixels[:, 1] - np.vectorize(eval_func_at_x)(pixels[:, 0]), scale))
    length = len(deviations)

    for i in range(1, divisor - 3):
        lower_b = length * i // divisor
        upper_b = length * (i+3) // divisor

        devs = np.divide(np.sum(deviations[lower_b : upper_b]),(upper_b - lower_b))

        if devs >  1 :
            return False, f"We were unsure if the shape of your graph is correct, particularly between x={round(pixels[lower_b][0], 1)} and x={round(pixels[upper_b][0], 1)}. Try using the guide points to help draw a smooth curve.\n<br>"
    
    return True, "We checked the overall shape of your graph, and it seems correct!\n<br>"

def deviations_check(pixels, eval_func_at_x, percentage, bound, check_sum_squares=True):
    devs = np.abs(pixels[:, 1] - np.vectorize(eval_func_at_x)(pixels[:, 0]))
    deviations = np.sort(devs)
    # Ensure that the sum of least squares of deviations from correct value is within a certain error bound
    sum_squares = 0
    sum_squares = sum(map(lambda dev : dev ** 2, deviations))
    feedback = ""
    if check_sum_squares and not (sum_squares / len(deviations)) < SQUARES_ERROR_BOUND:
        feedback = f"Coordinates are outside of error bounds. Sum of squares = {sum_squares / len(pixels)}\n<br>"
    if not deviations[(len(deviations) * percentage) // 100] < bound:
        feedback += f"Too many points are outside a reasonable range of the function.\n<br>"
    return (not check_sum_squares or (sum_squares / len(deviations)) < SQUARES_ERROR_BOUND) and deviations[(len(deviations) * percentage) // 100] < bound, feedback


# sliding check for multiple y values per x value - requires at least 3 squares
def one_to_many_check(pixels, eval_func_at_x, percentage, num_squares):
    n = len(pixels)
    for i in range(1, num_squares - 3):
        lower_b = n * i // num_squares
        upper_b = n * (i+3) // num_squares
        # find percentage of x values with multiple y values
        curr = pixels[lower_b]
        occurrences = 0
        for j in range(lower_b, upper_b - 1):
            if abs(pixels[j+1][1] - pixels[j][1]) > 0.1:
                occurrences += 1
        proportion = occurrences / n
        if proportion > percentage:
            return False, f"We were unable to correctly classify the graph. Please check behaviour around ... {proportion}"
    return True, f"One to many passed"

def gradient_check(pixels, eval_func_at_x):
    expected_slope = eval_func_at_x(1) - eval_func_at_x(0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(pixels[:, 0], pixels[:, 1])
    if np.abs((slope - expected_slope) / expected_slope) < RELATIVE_GRADIENT_ERROR:
        return True, "We checked the gradient and it is correct\n<br>"
    else:
        return False, f"We expected a gradient of {expected_slope}, but your graph has a gradient of about: {slope}\n<br>"

def pythagorean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def critical_point_check(pixels, point, axis_scales, compare_axis=0, mode='intercept'):
    """
    Checks if a point exists within a certain distance of a given critical point.
    ---
    Note: compare_axis refers to the axis which we want to fix.
    For example in checking the y-intercept, we want to fix the x-coordinate at 0 and check how far off the y-coordinate,
    and for checking the x-intercept, we want the converse.
    This can be used to check maxima and minima points of polynomials as well.

    Args:
        pixels (numpy.ndarray): array of coordinates of the student drawn pixels
        point (__type__): the critical point we want to check the distance of
        axis_scales (float, float): the scales of the (x, y) axes
        compare_axis (int, optional): the axis we want to fix. Defaults to 0.
        mode (str, optional): the mode, can be maxima, minima or intercept

    Returns:
        bool: Did the check succeed.
    """
    PERCENTILE = 40
    other_axis = 1 - compare_axis
    critical_region = pixels[ np.abs(pixels[:, compare_axis] - point[compare_axis]) < CRITICAL_POINT_TOLERANCE * axis_scales[compare_axis] ]

    if np.size(critical_region) == 0:
        return False
    elif mode == 'maxima':
        threshold = np.percentile(critical_region[:, other_axis], 100 - PERCENTILE)
        critical_region = critical_region[ critical_region[:, other_axis] >  threshold ]
    elif mode == 'minima':
        threshold = np.percentile(critical_region[:, other_axis], PERCENTILE)
        critical_region = critical_region[ critical_region[:, other_axis] <  threshold ]

    fold_func = np.any if mode == 'intercept' else np.all
    return fold_func(np.abs(critical_region[:, other_axis] - point[other_axis]) < CRITICAL_POINT_TOLERANCE * axis_scales[other_axis])

def y_intercept_check(pixels, eval_func_at_x, x_scale, y_scale):
    if critical_point_check(pixels, (0, eval_func_at_x(0)), (x_scale, y_scale)):
        return True, f"You correctly found the y intercept at {(0, round(eval_func_at_x(0), 1))}.\n<br>"
    else:
        return False, "We tested your y-intercept and it was incorrect.\n<br>"

def real_roots_check(pixels, eval_func_at_x, y_scale, x_scale):
    x_intercepts = [ (float(root), 0) for root in real_roots(eval_func_at_x) ]
    if len(x_intercepts) == 0:
        return True, ""
    x_ints = all([ critical_point_check(pixels, x_intercept, (x_scale, y_scale), compare_axis=1) for x_intercept in x_intercepts])
    x_ints_fb = "You've found all the roots of the equation and drawn them correctly.\n<br>" if x_ints else f"Looks like you're missing at least one root - have you tried equating the polynomial to 0 and solving for x?\n<br>"
    return x_ints, x_ints_fb 

def check_maxima(pixels, maximas, x_scale, y_scale):
    if maximas == []:
        return True, ""
    for maxima in maximas:
        if not critical_point_check(pixels, maxima, (x_scale, y_scale), mode='maxima'):
            return False, f"Looks like you're missing at least one maxima - have you tried differentiating the equation and finding the roots?\n<br>"
    return True, f"You've correctly identified all maxima\n<br>"

def check_minima(pixels, minimas, x_scale, y_scale):
    if minimas == []:
        return True, ""
    for minima in minimas:
        if not critical_point_check(pixels, minima, (x_scale, y_scale), mode='minima'):
            return False, f"Looks like you're missing at least one minima - have you tried differentiating the equation and finding the roots?\n<br>"
    return True, "You have correctly identified all minima\n<br>"

def x_intercept_check(pixels, eval_func_at_x, x_scale, y_scale):
    if critical_point_check(pixels, (- eval_func_at_x(0) / (eval_func_at_x(1) - eval_func_at_x(0)), 0), (x_scale, y_scale), compare_axis=1):
        return True, "You have correct x-intercept!\n<br>"
    else:
        return False, "Double check your x-intercepts.\n<br>"

def check_dom_coeff(pixels, eval_func_at_x):
    expected_coeff = eval_func_at_x.coeffs()[0]
    deg = degree(eval_func_at_x)
    observed_coeff = np.polyfit(pixels[:, 0], pixels[:, 1], deg)[0]

    expected_coeff_sgn = np.sign(expected_coeff)
    observed_coeff_sgn = np.sign(observed_coeff)

    if expected_coeff_sgn == observed_coeff_sgn:
        return True, ""
    elif expected_coeff_sgn == 1 and observed_coeff_sgn == -1:
        return False, "How should the graph behave at the endpoints?\n<br>"
        # return False, "Expected positive dominant coefficient but found negative\n"
    else:
        return False, "How should the graph behave at the endpoints?\n<br>"

def check_additional_intercepts(pixels, intercepts):
    observed_ints = list(filter(lambda coord: abs(coord[0]) < 0.0001 or abs(coord[1]) < 0.0001, pixels))
    for o_int in observed_ints:
        close = False
        for intercept in intercepts:
            close = close or pythagorean_dist(o_int, intercept) < 1
        if not close:
            return False, "Double check where your graph intercepts the axis.\n<br>"
    return True, ""

def check_additional_turning_pts(pixels, eval_func_at_x, turning_pts, params):
    x_smoothed = savgol_filter(pixels[:, 0], 100, 3).reshape((-1, 1))
    y_smoothed = savgol_filter(pixels[:, 1], 100, 3).reshape((-1, 1))
    pixels = np.hstack((x_smoothed, y_smoothed))
    width = 90
    for i in range(50, len(pixels) - 140, 30):
        if ((pixels[i+width][1] - pixels[i][1]) * (eval_func_at_x(pixels[i+width][0]) - eval_func_at_x(pixels[i][0])) < 0 and
            abs(eval_func_at_x(pixels[i][0]) - eval_func_at_x(pixels[i+width][0])) > 0.25 * (params["y_scale"] / params["x_scale"]) and
            abs(diff(eval_func_at_x)(pixels[i+(int(width // 2))][0])) < 10 * (params["y_scale"] / params["x_scale"]) and
            all([tp[0] < pixels[i][0] or tp[0] > pixels[i+width][0] for tp in turning_pts])):
            return False, f"Did you mean to include a turning point at ({round(pixels[i][0], 1)}, {round(pixels[i][1], 1)})?\n<br>"
    
    for i in range(140, len(pixels) - 50, 30):
        if ((pixels[i-width][1] - pixels[i][1]) * (eval_func_at_x(pixels[i-width][0]) - eval_func_at_x(pixels[i][0])) < 0 and
            abs(eval_func_at_x(pixels[i][0]) - eval_func_at_x(pixels[i-width][0])) > 0.25 * (params["y_scale"] / params["x_scale"]) and
            abs(diff(eval_func_at_x)(pixels[i+(int(width // 2))][0])) < 10 * (params["y_scale"] / params["x_scale"]) and
            all([tp[0] < pixels[i][0] or tp[0] > pixels[i-width][0] for tp in turning_pts])):
            return False, f"Did you mean to include a turning point at ({round(pixels[i][0], 1)}, {round(pixels[i][1], 1)})?\n<br>"
    
    return True, ""

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
    coords = [coeff[0], -coeff[1]] * coords
    coords += [params["x_lower"] - params["x_scale"], params["y_upper"] + params["y_scale"]]

    def moving_average(a, n=10):
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    x_smoothed = moving_average(coords[:, 0]).reshape((-1, 1))
    y_smoothed = moving_average(coords[:, 1]).reshape((-1, 1))
    #x_smoothed = savgol_filter(coords[:, 0], 100, 3).reshape((-1, 1))
    #x_smoothed = coords[:, 0].reshape((-1, 1))
    #y_smoothed = savgol_filter(coords[:, 1], 100, 3).reshape((-1, 1))

    #import matplotlib.pyplot as plt
    #plt.plot(coords[:, 0], coords[:, 1])
    #plt.plot(x_smoothed, y_smoothed, color='red')
    #plt.savefig("fig3")

    return np.hstack((x_smoothed, y_smoothed))
