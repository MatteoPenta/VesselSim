# Background Generation
Part of the dataset includes empty masks to generate background images (aka no vessels) so that a deep learning model 
knows that not all patches will contain vessels. In our experiments, we created 2500 empty masks for training and an 
additional 250 masks for validation. 

An empty mask sample is available in this folder and is already formatted to function with the domain-randomized 
rendering pipeline introduced in [VesselFM](https://github.com/bwittmann/vesselFM): _D<sub>drand</sub>_. Please refer to 
their code and instructions to generate the angiographic image volumes.