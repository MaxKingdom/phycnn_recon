# phycnn_recon
This repo is the official implementation of "Displacement reconstruction using a physics-informed convolutional neural network with multiple residual 
autoencoder blocks"

## Updates
2022/6/2: First time of submit, main codes of the network

## Abstract
Deflection is an essential index to evaluate the operating state of a bridge but of difficult to measure, since it requires a stationary reference. This paper proposes an indirect method to measure the displacement time history. A convolutional neural network (CNN) is adopted to approximate the mapping relationship among bridge responses, with which the required deflec-tions are reconstructed from other measurements. The network architecture design is guided by prior knowledge derived from the traffic flow-induced bridge responses and the displacement reconstruction problem. The network has two individual branches to compute the quasi-static and dynamic displacement components, and sums them up in a later stem. Besides, the loss function involves a physics-based regularization term, i.e. the calculus relationship between displacement and acceleration. The physical loss can guide the training direction, alleviate overfitting issues, and improve the algorithmic performance on the high-frequency responses. Two numerical examples and a laboratory test are adopted to validate the performance and ap-plicability of the proposed approach.

## Authors
Peng Ni, Yixian Li, Limin Sun, and Ao Wang<dr>
Bridge Dep. of Tongji Uni.
