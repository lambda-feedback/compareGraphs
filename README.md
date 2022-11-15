# Evaluation Function Template Repository

This template repository contains the boilerplate code needed in order to create an AWS Lambda function that can be written by any tutor to grade a response area in any way they like.

This version is specifically for python, however the ultimate goal is to make similar boilerplate repositories in any language, allowing tutors the freedom to code in what they feel most comfortable with.

## Table of Contents
- [Evaluation Function Template Repository](#evaluation-function-template-repository)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Usage](#usage)
    - [Getting Started](#getting-started)
  - [How it works](#how-it-works)
    - [Docker & Amazon Web Services (AWS)](#docker--amazon-web-services-aws)
    - [Middleware Functions](#middleware-functions)
    - [GitHub Actions](#github-actions)
  - [Pre-requisites](#pre-requisites)
  - [Contact](#contact)

## Repository Structure

```bash
app/
    __init__.py
    evaluation.py # Script containing the main evaluation_function
    docs.md # Documentation page for this function (required)
    evaluation_tests.py # Unittests for the main evaluation_function
    requirements.txt # list of packages needed for algorithm.py
    Dockerfile # for building whole image to deploy to AWS

.github/
    workflows/
        test-and-deploy.yml # Testing and deployment pipeline

config.json # Specify the name of the evaluation function in this file
.gitignore
```

## Usage

### Getting Started

1. Clone this repository
2. Change the name of the evaluation function in `config.json`
3. The name must be unique. To view existing grading functions, go to:

   - [Staging API Gateway Integrations](https://eu-west-2.console.aws.amazon.com/apigateway/main/develop/integrations/attach?api=c1o0u8se7b&region=eu-west-2&routes=0xsoy4q)
   - [Production API Gateway Integrations](https://eu-west-2.console.aws.amazon.com/apigateway/main/develop/integrations/attach?api=cttolq2oph&integration=qpbgva8&region=eu-west-2&routes=0xsoy4q)

4. Merge commits into the default branch
   - This will trigger the `test-and-deploy.yml` workflow, which will build the docker image, push it to a shared ECR repository, then call the backend `grading-function/ensure` route to build the necessary infrastructure to make the function available from the client app.

5. You are now ready to start developing your function:
   
   - Edit the `app/evaluation.py` file, which ultimately gets called when the function is given the `eval` command
   - Edit the `app/evaluation_tests.py` file to add tests which get run:
       - Every time you commit to this repo, before the image is built and deployed 
       - Whenever the `healthcheck` command is supplied to the deployed function
   - Edit the `app/docs.md` file to reflect your changes. This file is baked into the function's image, and is made available using the `docs` command. This feature is used to display this function's documentation on our [Documentation](https://lambda-feedback.github.io/Documentation/) website once it's been hooked up!

---

## How it works

The function is built on top of a custom base layer, [BaseEvaluationFunctionLayer](https://github.com/lambda-feedback/BaseEvalutionFunctionLayer), which tools, tests and schema checking relevant to all evaluation functions.

### Docker & Amazon Web Services (AWS)

The grading scripts are hosted AWS Lambda, using containers to run a docker image of the app. Docker is a popular tool in software development that allows programs to be hosted on any machine by bundling all its requirements and dependencies into a single file called an **image**.

Images are run within **containers** on AWS, which give us a lot of flexibility over what programming language and packages/libraries can be used. For more information on Docker, read this [introduction to containerisation](https://www.freecodecamp.org/news/a-beginner-friendly-introduction-to-containers-vms-and-docker-79a9e3e119b/). To learn more about AWS Lambda, click [here](https://geekflare.com/aws-lambda-for-beginners/).

### Middleware Functions
In order to run the algorithm and schema on AWS Lambda, some middleware functions have been provided to handle, validate and return the data so all you need to worry about is the evaluation script and testing.

The code needed to build the image using all the middleware functions are available in the [BaseEvaluationFunctionLayer](https://github.com/lambda-feedback/BaseEvalutionFunctionLayer) repository.

### GitHub Actions
Whenever a commit is made to the GitHub repository, the new code will go through a pipeline, where it will be tested for syntax errors and code coverage. The pipeline used is called **GitHub Actions** and the scripts for these can be found in `.github/workflows/`.

On top of that, when starting a new evaluation function, you will have to complete a set of unit test scripts, which not only make sure your code is reliable, but also helps you to build a _specification_ for how the code should function before you start programming.

Once the code passes all these tests, it will then be uploaded to AWS and will be deployed and ready to go in only a few minutes.

## Pre-requisites
Although all programming can be done through the GitHub interface, it is recommended you do this locally on your machine. To do this, you must have installed:

- Python 3.8 or higher.

- GitHub Desktop or the `git` CLI.

- A code editor such as Atom, VS Code, or Sublime.

Copy this template over by clicking **Use this template** button found in the repository on GitHub. Save it to the `lambda-feedback` Organisation.

## Contact
