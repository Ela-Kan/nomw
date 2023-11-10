# ⌚ Notch On My Watch ⌚
Part of the UCL Centre for Medical Image Computing's (CMIC) Hackathon 9-10th November 2023: [CMICHacks 2.0](https://cmic-ucl.github.io/CMICHACKS/)

## 🚀 Project Title
Unlocking BBB Insights with GANs and Autoencoders: Correcting Ktrans Values in DCE MRI for Enhanced Understanding of Inflammation-Related Brain Disorders

## 👥 Project Members
**Leader**: [Mara Cercignani](https://github.com/maracerci)	(CUBRIC), Cardiff University <br>
Team Members:
  * [Sophie Martin](https://github.com/sophmrtn) (CMIC)
  * [Ela Kanani](https://github.com/Ela-Kan) (CMIC)
  * [Gabriel Oliveira Stahl](https://github.com/GabrielStahl) (CMIC)
  * [Isaac Llorente Saguer](https://github.com/isaac-6) (CMIC)

## ✍ Project Description (Mara's Project Brief)
![Example of problem: A DCE voxel is show with its filtered and unfiltered signal.](https://cmic-ucl.github.io/CMICHACKS/images/project_MC.jpeg)

Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE MRI) is a technique used to assess the permeability of the blood-brain barrier (BBB). The parameter of interest, known as Ktrans, is typically estimated by fitting a mathematical model to the data acquired from serial T1-weighted images taken after the gadolinium injection. In our study, we have identified negative Ktrans values primarily attributed to filtering procedures applied by the MRI scanner. This project's core objective is to rectify these issues by learning the mapping between filtered and unfiltered data. We possess both filtered and unfiltered MRI data from the same acquisitions, which serves as the basis for our approach. To tackle this challenge, we would like to explore multiple avenues, including the possibility of training autoencoders or utilizing Generative Adversarial Networks (GANs) to map between filtered and unfiltered images.


## 💭 Approach
We have created a conditional generative adversarial network (CGAN) to map between filtered and unfiltered images, taking the filtered image as the input and the unfiltered image as the output for further analysis. The filter is dependent on the subject's anatomy, hence we have constrained by subject, where each subject (15) has 142 volumes. 

## 📁 File Structure
```
.
├── data/
│   └── MNI/
│       ├── subject_folder_n
│       └── ...
├── Data_Loader.py
├── Discriminator.py
├── generator.py
├── train.py
└── utils.py

```
