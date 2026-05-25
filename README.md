# compareGraphs

> **Deprecated:** This project is no longer maintained and has been deprecated. No further updates or support will be provided.

An AWS Lambda evaluation function that grades student-drawn graphs against a provided polynomial equation and returns detailed feedback.

Given a student's graph as a PNG (base64 URI) and a correct equation in [SymPy](https://docs.sympy.org/latest/index.html) format, the function normalises the pixel coordinates to the graph's axis space and checks the drawing against the expected curve. Different checks are applied depending on whether the equation is linear or a higher-order polynomial.

## Table of Contents
- [Repository Structure](#repository-structure)
- [How it works](#how-it-works)
  - [Linear polynomials](#linear-polynomials)
  - [Higher-order polynomials](#higher-order-polynomials)
- [Input/Output](#inputoutput)
- [Docker & AWS](#docker--aws)
- [GitHub Actions](#github-actions)

## Repository Structure

```
app/
    evaluation.py        # Main evaluation logic
    evaluation_tests.py  # Unit tests
    docs/
        user.md          # Usage documentation and examples
        dev.md           # Developer documentation
    requirements.txt
    Dockerfile

.github/
    workflows/
        test-and-deploy.yml

config.json              # Evaluation function name
```

## How it works

The student's response is a PNG URI of their drawn graph. The function uses [PIL](https://pillow.readthedocs.io/en/stable/index.html) to extract the drawn pixels, then applies a linear transformation to map pixel coordinates onto the graph's coordinate space using the axis bounds and scale provided in `params`.

### Linear polynomials

For straight-line graphs, the function checks:
- **Gradient** — compares the observed slope (via linear regression) against the expected slope
- **Y-intercept** — checks the drawing passes through the correct y-intercept
- **X-intercept** — checks the drawing passes through the correct x-intercept
- **Coverage** — the line must cover a sufficient portion of the graph area
- **Density** — no large gaps in the drawn line

### Higher-order polynomials

For curves, the function checks:
- **Shape** — a sliding-window deviations check across the full curve
- **Dominant coefficient sign** — whether the curve's end behaviour is correct
- **Y-intercept** — whether the curve passes through the correct y-intercept
- **X-intercepts / roots** — all real roots of the polynomial must be correctly drawn
- **Maxima and minima** — all turning points must be present and correctly positioned
- **Spurious features** — flags unexpected intercepts or extra turning points in the student's drawing

## Input/Output

**Request:**
```json
{
  "response": {
    "student_answer": "<PNG data URI of the student's drawn graph>",
    "params": {
      "x_upper": "<int>",
      "x_lower": "<int>",
      "y_upper": "<int>",
      "y_lower": "<int>",
      "x_scale": "<int>",
      "y_scale": "<int>",
      "error_bound": "<float, optional>"
    }
  },
  "answer": "<equation string in SymPy format, e.g. \"-2*x+1\">"
}
```

The `answer` must use SymPy-compatible syntax — `"-2*x+1"` is valid, `"-2x+1"` is not.

**Response:**
```json
{
  "is_correct": "<bool>",
  "feedback": "<string>"
}
```

## Docker & AWS

The function runs as a Docker container on AWS Lambda. The image is built from `app/Dockerfile` and pushed to a shared ECR repository on each merge to the default branch.

## GitHub Actions

The `.github/workflows/test-and-deploy.yml` pipeline runs unit tests on every commit and, if they pass, builds and deploys the Docker image to AWS.

## Contact