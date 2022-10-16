
# Diffusion Models

Basic implementation of Diffusion models using the algorithms mentioned in the original DDPM paper. Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).


<img width=600 src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/unCLIP.png">
