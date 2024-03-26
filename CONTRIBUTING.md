# Contributing

## Developer Guide

### Setup

This project uses [hatch](https://hatch.pypa.io/latest/) to manage the development environment and build and publish releases. Make sure `hatch` is [installed](https://hatch.pypa.io/latest/install/) first:

```bash
$ pip install hatch
```

Now you can [enter the development environment](https://hatch.pypa.io/latest/environment/#entering-environments) using:

```bash
$ hatch shell
```

This will install development dependencies in an isolated environment and drop you into a shell (use `exit` to leave).

### Pre-commit

Use [pre-commit](https://pre-commit.com/) to run linting, type-checking, and formatting:

```bash
$ hatch run pre-commit run --all-files
```

...or install it to run automatically before every commit with:

```bash
$ hatch run pre-commit install
```

### Testing

Unit tests are _not_ run by `pre-commit`, but can be run manually using `hatch` [scripts](https://hatch.pypa.io/latest/config/environment/overview/#scripts):

```bash
$ hatch run test:all
```

Measure test coverage with:

```bash
$ hatch run test:coverage
```
