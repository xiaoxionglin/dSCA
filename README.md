# dSCA

demixed Sparse Component Analysis.

demix the neural signal elicited by different task variables e.g. (Stim1, Stim2, Time/context)

Sparse component analysis allow dimensionality reduction while taking into account the constraint of biological implementation (neurons have small number of synapses compared to the number of neurons in a cortical region). This will usually also yield small groups of dominant neurons.
In another interpretation, SCA finds the response profile of the hypothetical sources.


prerequisite:

- tensorflow2





technical note:

for some reason tensorflow2 doesn't work when directly used in jupyter .ipynb file, but works when wrapped in a function like in the demo.
in ipython console directly using tensorflow is ok but not the wrapped function.
