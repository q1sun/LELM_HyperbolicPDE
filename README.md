# LELM_HyperbolicPDE

Deep learning methods, which exploit auto-differentiation to compute derivatives without dispersion or dissipation errors, have recently emerged as a compelling alternative to classical mesh-based numerical schemes for solving hyperbolic conservation laws. However, solutions to hyperbolic problems are often piecewise smooth, posing challenges for training of neural networks to capture solution discontinuities and jumps across interfaces. 

In this paper, we propose a novel lift-and-embed learning method to effectively resolve these challenges. The proposed method comprises three innovative components: 

-  embedding the Rankine-Hugoniot condition within a one-order higher-dimensional space by including an augmented variable;
-  utilizing neural networks to handle the increased dimensionality and address both linear and nonlinear problems within a unified mesh-free learning framework; 
-  projecting the trained model back onto the original physical domain to obtain the approximate solution.

Notably, the location of discontinuities also can be treated as trainable parameters in our method and inferred concurrently with the training of neural network solutions. With collocation points sampled only on piecewise surfaces rather than fulfilling the whole lifted space, we demonstrate through extensive numerical experiments that our method can efficiently and accurately solve scalar hyperbolic equations with discontinuous solutions without spurious smearing or oscillations.

| Equation | Section | Neural Network (Depth, Width) | Penalty Coeff. (β_S, β_B, β_I) | Train Data Size (N_Intrr, N_Shock, N_Bndry, N_Initl) | Test Data Size (N_x, N_t) | Training Epochs | Runtime (h:m:s) | # Trainable Parameters | Memory (MB) | Learning Rate (solution, speed) |
|----------|---------|-------------------------------|---------------------------------|--------------------------------------------------------|----------------------------|----------------|----------------|----------------------|-------------|--------------------------------|
| Linear Convection | 3.1.1 | (2, 40) | (400, 1, 400) | (10k, 4k, 2k, 1k) | (1k, 1k) | 12k | 0:18:50 | 3,481 | 13.60 | (1e-2, ✗) |
| Linear Convection | 4.1.1 | (2, 40) | (400, 1, 400) | (10k, 4k, 2k, 1k) | (1k, 1k) | 12k | 0:20:37 | 3,481 | 13.60 | (1e-2, ✗) |
| Linear Convection | 4.1.2 | (6, 40) | (400, 10, 400) | (80k, 60k, 5k, 5k) | (1k, 1k) | 25k | 2:04:41 | 5,121 | 20.00 | (1e-2, ✗) |
| Inviscid Burgers | 3.1.2 | (3, 40) | (400, 1, 400) | (10k, 1k, 1k, 1k) | (1k, 1k) | 5k | 0:13:59 | 5,490 | 21.44 | (1e-2, ✗) |
| Inviscid Burgers | 4.2.1 | (6, 40) | (400, 1, 400) | (80k, 15k, 5k, 5k) | (1k, 1k) | 21k | 2:22:49 | 5,121 | 20.00 | (1e-2, ✗) |
| Inviscid Burgers | 4.2.2 | (6, 40) | (400, 1, 400) | (80k, 10k, 10k, 5k) | (1k, 1k) | 15k | 1:36:20 | 3,841 | 15.00 | (1e-2, ✗) |
| Inviscid Burgers | 4.2.3 | (4, 80) | (50, 1, 400) | (80k, 20k, 30k, 10k) | (1k×1k, 17) | 15k | 2:11:35 | 13,441 | 52.36 | (1e-2, ✗) |
| Inviscid Burgers | 4.3.1 | (4, 40) | (400, 1, 400) | (10k, 1k, 1k, 1k) | (1k, 1k) | 10k | 2:00:44 | 3,842 | 15.01 | (1e-2, 1e-2) |
| Inviscid Burgers | 4.3.2 | (6, 40) | (50, 1, 400) | (80k, 5k, 5k, 5k) | (1k, 1k) | 21k | 2:08:26 | 5,147 | 20.09 | (1e-2, 1e-2) |
| Buckley Equation | 4.4 | (6, 40) | (10, 1, 400) | (40k, 1k, 1k, 1k) | (1k, 1k) | 21k | 1:12:50 | 5,122 | 20.01 | (1e-2, 1e-3) |
| Euler Equations | 4.5 | (6, 40) | (20, 0, 40) | (80k, 5k, 0, 5k) | (1k, 1k) | 22k | 3:18:00 | 5,205 | 20.34 | (1e-2, 1e-2) |


## Citation

    @article{sun2024lift,
      title={Lift-and-Embed Learning Method for Solving Scalar Hyperbolic Equations with Discontinuous Solutions},
      author={Qi Sun, Zhenjiang Liu, Lili Ju, and Xuejun Xu},
      journal={arXiv preprint arXiv:2411.05382},
      year={2024}
    }
