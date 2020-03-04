pynnmap
=======

pynnmap is a library/application for creating nearest-neighbors vegetation
maps.  At present, it represents an in-house version of running
`Gradient Nearest Neighbor (GNN) <http://lemma.forestry.oregonstate.edu>`__
models, but we are working toward an application that can run a full suite
of nearest-neighbors techniques such as kNN, MSN, RFNN and others.  

As such, it borrows much of its inspiration from the R package
`yaImpute <https://cran.r-project.org/web/packages/yaImpute/>`__, but is based
on Python packages such as
`numpy <http://www.numpy.org/>`__,
`cython <http://cython.org/>`__,
`rasterio <https://github.com/mapbox/rasterio>`__ and
`pandas <http://pandas.pydata.org/>`__.

At this time, pynnmap is definitely not ready for prime-time.  We're stumbling
along as we go, trying to incorporate best practices.  If you're interested
in using this library, please let us know.  We welcome the interest.  If
you're interested in helping with its development, even better!
