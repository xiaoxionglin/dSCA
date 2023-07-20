# dSCA

demixed Sparse Component Analysis.

It consists of two parts: demixing and sparse component analysis.

Demixing means separating the neural signal elicited by different task variables e.g. (Stim1, Stim2, Time/context)

Sparse component analysis allow dimensionality reduction while taking into account the constraint of biological implementation (neurons have small number of synapses compared to the number of neurons in a cortical region). This has the advantage of giving specific and meaningful components unlike PCA where it's just a set of bases ranked by their eigenvalue. 

This will usually also yield small groups of dominant neurons, which serves as a mid-way between clustering and dim. reduction techniques.

If your data doesn't have single neurons, then consider this other interpretation: SCA finds the hypothetical source response profile


File description:
- dSCAdemo.ipynb
demo for the pipeline. dependent on other files.
- dSCA.py
it the dSCA method, to be imported
- demo/
a folder to store the grid search result in the demo
- 



prerequisite:

- tensorflow2





