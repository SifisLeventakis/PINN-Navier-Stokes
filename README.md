# PINN for 2D Incompressible Navier–Stokes (Lid-Driven Cavity)

## Problem Description
This project implements a Physics-Informed Neural Network (PINN) to solve the 2D incompressible Navier–Stokes equations for the lid-driven cavity flow at Re = 100. The governing equations are embedded directly into the loss function using  automatic differentiation.

<img width="380" height="400" alt="Screenshot 2026-03-27 113406" src="https://github.com/user-attachments/assets/a80a073e-e984-41a4-8ee5-8ed18010e28d" />

The governing equations of this 2D flow are the continuity and momentum equations in the x and y direction:

**Continuity Equation (mass conservation):**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**Momentum Equations (Momentum Conservation):**

| Direction | Equation |
| :--- | :--- |
| **X-Momentum** | $$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{Re} \nabla^2 u$$ |
| **Y-Momentum** | $$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{Re} \nabla^2 v$$ |

The computational domain of the problem is a unit square [0,1] x [0,1]. The boundary conditions of the four sides are:
* **Top moving wall:** $u(x,1) = 1$,  $v(x,1) = 0$
* **Bottom wall:** $u(x,0) = 0$,  $v(x,0) = 0$
* **Side walls:** $u(0,y) = 0$,  $v(0,y) = 0$ and $u(1,y) = 0$,  $v(1,y) = 0$

The training objective is to minimize a composite loss function that balances the governing equations with the boundary conditions. This is defined as:  
$$L_{total} = \omega_{pde} L_{physics} + \omega_{bc} L_{bc}$$

For the internal domain, we define an MSE loss function for 10000 collocation points, that enforces the Navier-Stokes equations:  
$$L_{physics} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left( |f_{cont, i}|^2 + |f_{u, i}|^2 + |f_{v, i}|^2 \right)$$

For the boundaries, we enforce the no-slip and lid-driven conditions at 1000 points and the boundary loss is as follows:  
$$L_{bc} = \frac{1}{N_b} \sum_{j=1}^{N_b} \left( |u_j - u_{target}|^2 + |v_j - v_{target}|^2 \right)$$

## Methodology
After trial and error below is the final setup of this study:  
* Neural network depth: 5
* Neural network width: 64
* Neural network activation function: tanh (infinetely differentiable, needed for second order derivatives)
* Starting with Adam optimizer and continuing with L-BFGS to increase accuracy of the result. L-BFGS is a second order optimization algorithm, thus it can further reduce the loss.
* We use a weight of 10 for the boundary loss compared to the physics loss (internal domain), since the network was found to struggle more with respecting the boundary conditions.

The standard lid-driven cavity problem contains mathematical singularities at the upper corners $(0,1)$ and $(1,1)$, where the moving lid meets the stationary side walls. To prevent the PINN from attempting to minimize an infinite gradient - which leads to numerical instability, a spatial weighting function (Smoothing Filter) was applied to the top boundary loss.  

Instead of a "hard" jump from $0$ to $1$, we apply a weight $\lambda(x)$ that ramps the lid velocity contribution:  
$$\lambda(x) = \max(0, 1 - 2|x - 0.5|)$$

The modified loss for the top boundary is calculated as:  
$$L_{top} = \frac{1}{N} \sum \lambda(x) \cdot \left( (u - 1)^2 + v^2 \right)$$

This is a way to tell the optimizer to care more about errors encountered close to the center of the top boundary and less close to the corners. This significantly decreased both the boundary and the physics losses by more than an order magnitude.

## Results
Below we can have insight on the results obtained. Starting with the different loss values, The loss decreases steadily during Adam training, with a significant additional drop upon switching to L-BFGS, demonstrating the benefit of the hybrid optimization strategy.

<img width="1899" height="612" alt="image" src="https://github.com/user-attachments/assets/90109c81-1be5-4fcd-ae46-7d2ba9915bdb" />

Below we can have insight on the results obtained. Starting with the loss history, the loss decreases steadily during Adam training, with a significant additional drop upon switching to L-BFGS, demonstrating the benefit of the hybrid optimization strategy.
The velocity magnitude contour is depicted below, showing the expected qualitative behavior. The primary vortex is located at approximately (0.62, 0.74). To validate quantitatively, the vortex center was identified by minimizing the velocity magnitude. The predicted center was found at (0.6263, 0.7475), compared to the Ghia et al. (1982) benchmark value of (0.6172, 0.7344), corresponding to a distance error of 0.01591.

<img width="400" height="380" alt="image" src="https://github.com/user-attachments/assets/d0f5a4fa-584d-4175-8dba-4922f385a742" />

Another important finding is the fluid streamlines in the cavity illustrated below. These results agree well with the literature:
* Smooth streamlines throughout the domain
* Primary vortex agrees with benchmark results
* Secondary vortices forming in the bottom left and right corners, as expected for Re=100 laminar flow

<img width="400" height="392" alt="image" src="https://github.com/user-attachments/assets/f04c8e0a-d397-49cc-b7ac-7a2d2db32791" />

To further validate the results, the u and v velocity components were extracted along the vertical and horizontal centerlines respectively, and compared against the Ghia et al. (1982) benchmark data. The u velocity profile shows excellent agreement with the reference solution along the entire vertical centerline (x=0.5). The v velocity profile captures the correct qualitative behavior but shows some discrepancy in the range x ∈ [0.15, 0.85], likely attributable to the corner singularities at the lid edges affecting the interior solution. The vorticity distribution along the moving lid also agrees well with the benchmark.

<img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/699cda3f-34e8-4fe2-a9f0-5bd768593304" />

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/aea6ed6a-8c73-4b9b-9368-4e0038f1d6c3" />

<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/0c1d3a3c-857a-472d-b7d2-5783e1d968d2" />

## Limitations & Future Work
The current implementation solves the lid-driven cavity at a fixed Reynolds number (Re=100). Known limitations include:

* Corner singularities at the lid edges introduce local errors in the boundary loss that affect the v velocity prediction in the interior
* Training is computationally expensive compared to classical CFD solvers for a single Re case

Future extensions planned:

* Parametric PINN — add Re as an additional network input, enabling instant flow field prediction across Re=100 to Re=1000 without retraining
* Adaptive sampling — concentrate collocation points in high-gradient regions (boundary layer, vortex core) to improve accuracy
* Higher Re — extend to Re=400 and Re=1000 where secondary vortices are more pronounced


## References

Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics, 48(3), 387-411.  

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

