# ViaSHAP regression

This directory contains the implementation of the paper "ViaSHAP: A SHAP-based approach for explaining black-box models in the presence of multiple features" by [Amr Alkhatib et al. (2025)](https://arxiv.org/pdf/2505.04775).

# Running Tests

This directory contains tests for the viashap_paper implementation. To run the tests, you'll need to ensure Python can find the source modules by adding the `src` directory inside the `viashap_paper` directory via the Python path.

First, you need to be located in the main repository for consistency (paper_implementations). Make sure you are there by running the following command on the terminal:

```bash
cd path/to/repository/paper_implementations
```

## Running All Tests

To run all tests in the test suite:

```bash
PYTHONPATH=viashap_paper/src pytest
```

## Running Specific Tests:

```bash
PYTHONPATH=viashap_paper/src pytest relative/path/from/paper_implementations/to/file
```

## Acknowledgements

- The KANLinear implementation is adapted from the work shown in this repository [Efficient KAN](https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py). I haven't implemented this code myself, and my hope is that the users of this project will just implement the network they want themselves instead of using the helper ones provided to be used for plug-and-play testing. But all credit goes to the maintainers of that repo for the KANLinear code.