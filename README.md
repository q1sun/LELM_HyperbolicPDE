# LELM_HyperbolicPDE

Deep learning methods, which exploit auto-differentiation to compute derivatives without dispersion or dissipation errors, have recently emerged as a compelling alternative to classical mesh-based numerical schemes for solving hyperbolic conservation laws. However, solutions to hyperbolic problems are often piecewise smooth, posing challenges for training of neural networks to capture solution discontinuities and jumps across interfaces. 

In this paper, we propose a novel lift-and-embed learning method to effectively resolve these challenges. The proposed method comprises three innovative components: 

(i)  (ii) and (iii)  
-  embedding the Rankine-Hugoniot condition within a one-order higher-dimensional space by including an augmented variable;
-  utilizing neural networks to handle the increased dimensionality and address both linear and nonlinear problems within a unified mesh-free learning framework; 
-  projecting the trained model back onto the original physical domain to obtain the approximate solution.

Notably, the location of discontinuities also can be treated as trainable parameters in our method and inferred concurrently with the training of neural network solutions. With collocation points sampled only on piecewise surfaces rather than fulfilling the whole lifted space, we demonstrate through extensive numerical experiments that our method can efficiently and accurately solve scalar hyperbolic equations with discontinuous solutions without spurious smearing or oscillations.

## Citation

    @article{sun2024lift,
      title={Lift-and-Embed Learning Methods for Solving Scalar Hyperbolic Equations with Discontinuous Solutions},
      author={Qi Sun, Zhenjiang Liu, Lili Ju, and Xuejun Xu},
      journal={arXiv preprint arXiv:2411.05382},
      year={2024}
    }
