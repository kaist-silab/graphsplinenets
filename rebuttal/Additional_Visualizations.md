
# GraphSplineNets

## Model Components Running Time Analysis

We tested and visualized the running time of the different components of our model, compare with the MGN baseline model. The following figure illustrates the results:

<div align="center">
    <img src="assets/running-time.png"/>
</div>

The running time components for each model are as follows:
- `Baseline` models take all sample points as input and do not require any upsampling step. Therefore, the neural network's running time for these models is equivalent to the total running time;
- `MGN+OSC(Post)` and `MGN+OSC`: MGN inference time + OSC upsampling time;
- `MGN+OSC+Adaptive`: MGN inference time + OSC upsampling time + collocation points adaptation time;

We can notice that the OSC costs minor computation compared to the network inference time thanks to efficient implementations. Since the MGN+OSC(Post) doesn't need to be differentiable, we can run this model in CPU or GPU. As discussed in Figure.3 in the paper, running in GPU will save around 50% time of the OSC.

## 3D Visualization to the Wave Experiment

We visalize the results of the final prediction frame in the Wave experiment in 3D. The following figure illustrates the results:

<div align="center">
    <img src="assets/wave-3d.png"/>
</div>

This figure compares the simulation performance of the MGN+OSC and the MGN+OSC+Adaptive model in the wave dataset with a significant peak in the domain. With the adaptive collocation points, the model is able to capture the peak more accurately, since the collocation points are adapted to the peak region as shown in the right figure. Except this region, the model is able to capture the general trend of the simulation in a better way. For example collocation points will adapt to the wave region to emphasis more about the part. 

## Additional visualizetion to the MGN+OSC(Post) model

We additionally visualize results of the MGN+OSC(Post) model in different datasets. The following figure illustrates the results:

<div align="center">
    <img src="assets/post-osc.png"/>
</div>

We can see that compare with the baseline model which has noise in the prediction, with the help of the OSC, the MGN+OSC(Post) model is able to predict a continous simulation which will eliminate the noise. To achieve a more accurate simulation result, we worked on fitting the OSC more closely with the ground truth. After optimizing with loss through the OSC and adaptive collocation points, we observed an improvement in simulation accuracy, as shown in the error plots. This demonstrates the effectiveness of our full GraphSplineNets method, which combines MGN, OSC, and adaptive collocation points.