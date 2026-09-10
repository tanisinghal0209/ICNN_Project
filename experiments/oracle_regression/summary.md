# Oracle-supervised ICNN diagnostic

The D=16/D=32 ICNN uses the baseline 3x128 Softplus architecture, clipping, Adam (lr=1e-3, betas=(0.5, 0.9)), batch size 256, and 2,000 requested iterations. Only the objective is changed to MSE(grad_x f(x), T*(x)).

| D | Method | L2-UVP (%) | Cosine | L2 Error | Final objective | Peak f-grad | Mean f-grad | NaN/Inf |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 16 | baseline two-potential training | 37.7563 | 0.7731 | 6.0410 | 10.712418 | 2685.7641 | 17.9116 | False |
| 16 | oracle MSE | 29.1391 | 0.8249 | 4.6299 | 0.282952 | 1.0765 | 0.5085 | False |
| 32 | baseline two-potential training | 50.7988 | 0.7644 | 16.2556 | 16.554657 | 12999.4749 | 94.3782 | False |
| 32 | oracle MSE | 42.5931 | 0.8083 | 13.6303 | 0.434887 | 1.3867 | 0.6840 | False |

Interpretation should use the complete trajectories and all three official transport metrics. Low oracle error with finite, materially calmer gradients supports the training-dynamics hypothesis; persistent oracle fitting failure without the second potential implicates the clipped ICNN setup as well.

The two objectives have different units, so their losses are not compared directly. The transport metrics and f-parameter gradient-norm trajectories are the controlled comparisons.
