# LELM_HyperbolicPDE

Deep learning methods, which exploit auto-differentiation to compute derivatives without dispersion or dissipation errors, have recently emerged as a compelling alternative to classical mesh-based numerical schemes for solving hyperbolic conservation laws. However, solutions to hyperbolic problems are often piecewise smooth, posing challenges for training of neural networks to capture solution discontinuities and jumps across interfaces. 

In this paper, we propose a novel lift-and-embed learning method to effectively resolve these challenges. The proposed method comprises three innovative components: 

-  embedding the Rankine-Hugoniot condition within a one-order higher-dimensional space by including an augmented variable;
-  utilizing neural networks to handle the increased dimensionality and address both linear and nonlinear problems within a unified mesh-free learning framework; 
-  projecting the trained model back onto the original physical domain to obtain the approximate solution.

Notably, the location of discontinuities also can be treated as trainable parameters in our method and inferred concurrently with the training of neural network solutions. With collocation points sampled only on piecewise surfaces rather than fulfilling the whole lifted space, we demonstrate through extensive numerical experiments that our method can efficiently and accurately solve scalar hyperbolic equations with discontinuous solutions without spurious smearing or oscillations.

\begin{table}
\begin{tabular}{ccccccc}
\toprule
\multicolumn{2}{c}{} &
\makecell{Neural Network\ (Depth, Width)} &
\makecell{Penalty Coeff.\ ($\beta_{\textnormal{S}}$, $\beta_{\textnormal{B}}$, $\beta_{\textnormal{I}}$)} &
\makecell{Train Data Size\ ($N_{\textnormal{Intrr}}$, $N_{\textnormal{Shock}}$, $N_{\textnormal{Bndry}}$, $N_{\textnormal{Initl}}$)} &
\makecell{Test Data Size \ ($N_x$, $N_t$)} \
\midrule
\multirow{3}{\parbox{1.8cm}{\centering Linear Convection Equations}}
& Sec. 3.1.1 & (2, 40) & (400, 1, 400) & (10$k$, 4$k$, 2$k$, 1$k$) & (1$k$, 1$k$) \ \cline{2-6}
& Sec. 4.1.1 & (2, 40) & (400, 1, 400) & (10$k$, 4$k$, 2$k$, 1$k$) & (1$k$, 1$k$) \ \cline{2-6}
& Sec. 4.1.2 & (6, 40) & (400, 10, 400) & (80$k$, 60$k$, 5$k$, 5$k$) & (1$k$, 1$k$)\
\midrule
\multirow{6}{\parbox{1.8cm}{\centering Inviscid Burgers Equations}}
& Sec. 3.1.2 & (3, 40) & (400, 1, 400) & (10$k$, 1$k$, 1$k$, 1$k$) & (1$k$, 1$k$) \ \cline{2-6}
& Sec. 4.2.1 & (6, 40) & (400, 1, 400) & (80$k$, 15$k$, 5$k$, 5$k$) & (1$k$, 1$k$) \ \cline{2-6}
& Sec. 4.2.2 & (6, 40) & (400, 1, 400) & (80$k$, 10$k$, 10$k$, 5$k$) & (1$k$, 1$k$) \ \cline{2-6}
& Sec. 4.2.3 & (4, 80) & (50, 1, 400) & (80$k$, 20$k$, 30$k$, 10$k$) & ($1k\times 1k$, 17) \ \cline{2-6}
& Sec. 4.3.1 & (4, 40) & (400, 1, 400) & (10$k$, 1$k$, 1$k$, 1$k$) & (1$k$, 1$k$) \ \cline{2-6}
& Sec. 4.3.2 & (6, 40) & (50, 1, 400) & (80$k$, 5$k$, 5$k$, 5$k$) & (1$k$, 1$k$) \
\midrule
Buckley Equation & Sec. 4.4 & (6, 40) & (10, 1, 400) & (40$k$, 1$k$, 1$k$, 1$k$) & (1$k$, 1$k$)\
\midrule
Euler Equations & Sec. 4.5 & (6, 40) & (20, 0, 40) & (80$k$, 5$k$, 0, 5$k$) & (1$k$, 1$k$) \
\bottomrule
\end{tabular}
\end{table}

## Citation

    @article{sun2024lift,
      title={Lift-and-Embed Learning Method for Solving Scalar Hyperbolic Equations with Discontinuous Solutions},
      author={Qi Sun, Zhenjiang Liu, Lili Ju, and Xuejun Xu},
      journal={arXiv preprint arXiv:2411.05382},
      year={2024}
    }
