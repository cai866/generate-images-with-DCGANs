# generate-images-with-DCGANs
The project is about generate images by DCGAN. The generator will generate fake images and it's goal is deceive the discriminator. However, the discriminator is going to discriminate all fake images from real images. By many trainings, 
the optimizor will minimize the loss functions of generator and discriminator. Finally, the generator will be able to produce qualified images.

![image](https://github.com/cai866/generate-images-with-DCGANs/blob/master/pictures/constructure.png)

Generator is an upsampling network with fractionally-strided convolutions

Discriminator is a convolutional network

Use batchnorm in both the generator and the discriminator
Use ReLU activation in generatot for all layers expects for the output, which uses Tanh
Use LeakyReLU activation in the discriminator for all layers

Optimizor is RMSprop algorithm.



![image](https://github.com/cai866/generate-images-with-DCGANs/blob/master/pictures/dcgan_demo.gif)

