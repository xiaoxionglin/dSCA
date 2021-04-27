# dSCA

demixed Sparse Component Analysis.

It consists of two parts: demixing and sparse component analysis.

Demixing means separating the neural signal elicited by different task variables e.g. (Stim1, Stim2, Time/context)

Sparse component analysis allow dimensionality reduction while taking into account the constraint of biological implementation (neurons have small number of synapses compared to the number of neurons in a cortical region). This has the advantage of giving specific and meaningful components unlike PCA where it's just a set of bases ranked by their eigenvalue. 

This will usually also yield small groups of dominant neurons, which serves as a mid-way between clustering and dim. reduction techniques.

If your data doesn't have single neurons, then consider this other interpretation: SCA finds the hypothetical source response profile


prerequisite:

- tensorflow2





technical note:

for some reason tensorflow2 doesn't work when directly used in jupyter .ipynb file, but works when wrapped in a function like in the demo.
in ipython console directly using tensorflow is ok but not the wrapped function.
