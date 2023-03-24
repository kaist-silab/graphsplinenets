# GraphSplineNets 

## Burgers Equation Experiment

We include in this file the additional experimental details for the new burgers equation dataset.

### Dataset Details
Burgers' equation or Bateman–Burgers equation is a fundamental partial differential equation and convection–diffusion equation occurring in various areas of applied mathematics, such as fluid mechanics, nonlinear acoustics, gas dynamics, and traffic flow. The equation was first introduced by Harry Bateman in 1915 and later studied by Johannes Martinus Burgers in 1948.

Burgers equation experiment uses a dataset directory which contains solutions to the time-dependent Burgers equation in one dimension. 

The Burgers equation has the form:

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

The Burgers equations is sometimes called "the poor man's Navier Stokes equation"; it can be regarded as a cousin of that equation, which still includes nonlinearity controlled by the magnitude of a viscosity parameter nu, and whose solution exhibits wave-like behavior.

### Training process
Since this is a one-dimension euqation, so we make a set of isometric partitions and isometricly select $r+1$ collocation points in each partition as described in the section 3.3 of the paper about the OSC. Then we create a fully connected graph on these collocation points and train the model. The dataset is availabel at the [following link](https://people.sc.fsu.edu/~jburkardt/datasets/burgers/burgers.html).

### Training Details
We train the model with the same hyperparameters as in the paper, as explained in Appendix B.5.

### Results
We visualize the results of the model on the test set. The following figure shows the ground truth and the predicted solution for the test set, also including two sepcific visualization at the initialization state and the value changes for one collocation point, shown as the space-oriented OSC performance and the time-oriented OSC performance.

<div align="center">
    <img src="https://anonymous.4open.science/r/graphsplinenets/rebuttal/assets/burger_equation.png"/>
</div>

We can see that the GraphSplineNets cannot handle the discontinuity fully, it can provide a smoothed-out approximation, resulting in a small overshoot in the solution. However, this result is not too bad overall. We would like to emphasize that our method assumes a certain degree of smoothness, but this holds for many scientific data, such as several instances of fluid simulation and weather forecasting. (For example, in Fourier Neural Operators (FNOs), an orthogonal direction to ours in surrogate modeling, the models also assume some level of smoothness, as Fourier Transform cannot represent discontinuities; unlike FNO, we can manage irregular meshes and boundaries).

---

We also include extended visualizations and explanations for other questions [here](Additional_Visualizations.md).