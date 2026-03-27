# PINN for 2D Incompressible Navier–Stokes (Lid-Driven Cavity)
This project implements a Physics-Informed Neural Network (PINN) to solve the 2D incompressible Navier–Stokes equations for the lid-driven cavity flow at Re = 100. The governing equations are embedded directly into the loss function using  automatic differentiation.

![Result](https://github.com/user-attachments/assets/ab2b75c9-4f33-473e-8bfa-88f42dea9f95)

The governing equations of this 2D flow are the continuity and momentum equations in the x and y direction:

**Continuity Equation (mass conservation):**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**Momentum Equations (momentum conservation):**
$$\begin{array}{lcl}
u \dfrac{\partial u}{\partial x} + v \dfrac{\partial u}{\partial y} & = & -\dfrac{\partial p}{\partial x} + \dfrac{1}{Re} \nabla^2 u \\
[10pt]
u \dfrac{\partial v}{\partial x} + v \dfrac{\partial v}{\partial y} & = & -\dfrac{\partial p}{\partial y} + \dfrac{1}{Re} \nabla^2 v
\end{array}$$

