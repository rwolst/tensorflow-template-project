# tensorflow-template
A standard Tensorflow project template that can be copied and modified for
new projects.

The example problem solves a softmax regression where the
independent variables are only known up to a multivariate normal distributions.
See the `latex/doc.pdf` for details on the mathematics.

The aim is to closely follow the best practices as set out
[here](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3).

## Installation
Recommend installing in a virtual enivronment. If you want to install
everything to your system environment then skip this step.

### Virtual Environment
In the project root directory, create virtual environment using `venv` (or a
similar alternative).

    python3 -m venv env

Activate the virtual environment.

    . env/bin/activate

### Install Dependencies
Do the this step whether in a virtual environment or not.

    python3 -m pip install -r requirements.txt

### Documentation
The documentation about the mathematics can be compiled into a `doc.pdf` by
running:

    cd latex
    pdflatex doc.tex

## Usage
First 'download' the data by running

    bash data/download.sh

Note: It doesn't actually download anything but instead generates the data. A
future project however may have this actually download something.

### Training
The following functions can be used

    python3 main.py Train --model-name SklearnModel --debug --max-iter 10

    python3 main.py Train --model-name TFModel --debug --max-iter 10

### Viualising

Using the main endpoint?

## Test
My test results?

## Useful Links
Scikit learn
Tensorflow
Tensorflow best practices
Softmax wiki
