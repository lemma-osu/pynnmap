# pynnmap

`pynnmap` is a library/application for creating nearest-neighbors vegetation maps. At present, it represents an in-house version of running [`Gradient Nearest Neighbor (GNN)`](https://lemma.forestry.oregonstate.edu) models, but we are working toward an application that can run a full suite of nearest-neighbors techniques such as kNN, MSN, RFNN and others.

As such, it borrows much of its inspiration from the R package [`yaImpute`](https://cran.r-project.org/web/packages/yaImpute/), but is based on Python packages such as [`numpy`](http://www.numpy.org/),
[`cython`](http://cython.org/), [`rasterio`](https://github.com/mapbox/rasterio), and [`pandas`](http://pandas.pydata.org/).

At this time, `pynnmap` is definitely not ready for prime-time. We're stumbling along as we go, trying to incorporate best practices. If you're interested in using this library, please let us know. We welcome the interest. If you're interested in helping with its development, even better!
