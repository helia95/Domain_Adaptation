# Domain adaptation
Experiments with one-source-one target domain adaptation. I tried to implement two papers, using as datasets MNISTM, USPS, SVHN. The results are reported below as the accuracy on the target domain, where each row represents the training domain, while the column represents a specific configuration of **source to target** (e.g. the value reported in (source, USPS to MNISTM) means the network has been trained *only* on USPS and evaluated of MNISTM, hence first and last rows should provide a lower and upper bound respectively).

## DANN
Implementation of Domain Adversarial training of Neural Network.

#### Results
|  | USPS to MNISTM | MNISTM to SVHN | SVHN to USPS |
| :--: | :--: | :--: | :--: |
| Source | 0.1845 | 0.28 | 0.5705 |
| DANN | 0.3024 | 0.4858 | 0.3687 |
| Target  | 0.9714 | 0.9137 | 0.9711

#### Latent space representation
2D representation (with t-SNE) of latent space of subset of source / target images.


   <p align="center">
  <img height = 500px, width=500px                   
   src=dann_svhn_to_usps.png>
  </p>

  <p align="center">
 <img height = 500px, width=500px                   
  src=dann_svhn_to_usps_1.png>
 </p>


## ADDA
Implementation of Adversarial Discriminative Domain Adaptation.

#### Results
|  | USPS to MNISTM | MNISTM to SVHN | SVHN to USPS |
| :--: | :--: | :--: | :--: |
| Source | 0.1845 | 0.28 | 0.5705 |
| ADDA | 0.605 | 0.1931 | 0.6138 |
| Target  | 0.9714 | 0.9137 | 0.9711

#### Latent space representation
2D representation (with t-SNE) of latent space of subset of source / target images.

  <p align="center">
  <img height = 500px, width=500px              
   src=adda_svhn_to_usps.png>
  </p>

  <p align="center">
 <img height = 500px, width=500px                   
  src=adda_svhn_to_usps_1.png>
 </p>
