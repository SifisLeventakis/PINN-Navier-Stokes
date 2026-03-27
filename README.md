# PINN for 2D Incompressible Navier–Stokes (Lid-Driven Cavity)
This project implements a Physics-Informed Neural Network (PINN) to solve the 2D incompressible Navier–Stokes equations for the lid-driven cavity flow at Re = 100. The governing equations are embedded directly into the loss function using  automatic differentiation.

![Result](https://github.com/user-attachments/assets/ab2b75c9-4f33-473e-8bfa-88f42dea9f95)

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




