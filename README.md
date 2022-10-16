
# Diffusion Models

Basic implementation of Diffusion models using the algorithms mentioned in the original DDPM paper. Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).


<img width=600 src="https://user-images.githubusercontent.com/45179700/196035228-50980dd3-ad88-4104-9f7e-215c7cd4146f.png">
