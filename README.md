# LELM_HyperbolicPDE

Machine learning methods, which exploit auto-differentiation to compute derivatives without dispersion or dissipation errors, have recently emerged as a compelling alternative to mesh-based numerical schemes for solving hyperbolic conservation laws. However, solutions to hyperbolic problems are often piecewise smooth, posing challenges for neural networks in capturing solution jumps across discontinuity interfaces. 

To effectively resolve the discontinuous solution of scalar hyperbolic equations, we propose lift-and-embed learning methods that compromise three innovative components: 

-  embedding the Rankine-Hugoniot condition within a higher-dimensional space by including an augmented variable;
-  utilizing neural networks to manage the increased dimensionality and address both linear and nonlinear problems within a unified mesh-free learning framework;
-  projecting the trained model back onto the original plane to obtain the approximate solution.

Notably, the location of discontinuities can be treated as trainable parameters and inferred concurrently with the training of network solution. With collocation points sampled on piecewise surfaces rather than fulfilling the lifted space, we demonstrate through numerical experiments that our methods achieve high-resolution of discontinuities without spurious smearing or oscillations.

## Citation

    @article{liu2024lift,
      title={Lift-and-Embed Learning Methods for Solving Scalar Hyperbolic Equations with Discontinuous Solutions},
      author={Zhenjiang Liu, Qi Sun, and Xuejun Xu},
      journal={arXiv preprint arXiv:2411.05382},
      year={2024}
    }
