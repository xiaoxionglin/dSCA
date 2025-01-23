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



# paper link

https://www.science.org/doi/full/10.1126/sciadv.adh8685

```
@article{lin2023neuronal,
  title={The neuronal implementation of representational geometry in primate prefrontal cortex},
  author={Lin, Xiao-Xiong and Nieder, Andreas and Jacob, Simon N},
  journal={Science Advances},
  volume={9},
  number={50},
  pages={eadh8685},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```
