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