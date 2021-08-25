Implementation of GPWL with both categorical Weisfeiler-Lehman (original WL) and continuous WL kernels.
The original WL is based on grakel implementation, and the continuous WL is built based on the methodology of Togninalli, M., Ghisu, E., Llinares-López, F., Rieck, B., & Borgwardt, K. (2019). Wasserstein weisfeiler-lehman graph kernels. Advances in Neural Information Processing Systems 33.

# Dependencies:
- grakel
- dgl
- networkx
- pytorch and gpytorch if using GP surrogates
- scipy and scikit-learn if using linear regression surrogate

Currently the input graphs are expected in dgl format, but it should be reasonably to convert to another popular format like networkx.
readm