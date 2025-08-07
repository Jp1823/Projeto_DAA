## Projeto_DAA

### Project Summary

This report is part of the practical work for the course on Data and Machine Learning, aimed at deepening the knowledge acquired throughout the semester.

The project is framed within a Kaggle challenge that seeks to analyze correlations in MRI images depending on the patient's development stage (Cognitively Normal - CN, Mild Cognitive Impairment - MCI, and Alzheimer's Disease - AD). Through a structured methodology, the provided dataset will be analyzed, explored, and preprocessed to extract relevant information about the problem.

This process enables the design and optimization of multiple Machine Learning models. The ultimate goal of this project is to develop a balanced model that leverages identified correlations and is capable of accurately predicting the stage of Alzheimer's disease progression in a patient.

---

Cognitive decline and dementia are increasingly prevalent worldwide, driven by a combination of genetic, lifestyle, and health-related factors.

#### Mild Cognitive Impairment (MCI)

Mild Cognitive Impairment (MCI) is a condition characterized by greater cognitive decline than expected for an individual's age and educational background, without significantly affecting daily activities.

Although MCI is less severe than dementia, it can progress to more serious conditions, particularly Alzheimer's Disease (AD). Because MCI represents a transitional state between normal aging and AD, early detection is essential to enable timely treatment and potentially reduce the number of dementia cases in the long term.

### MRI Imaging and Radiomics

Magnetic Resonance Imaging (MRI) allows the observation of structural brain changes associated with MCI, including:

- Shrinkage of the hippocampus
- Enlargement of fluid-filled brain regions
- Reduced glucose metabolism

Radiomics is a quantitative approach to medical imaging that extracts textural features from the spatial distribution and relationships of pixel or voxel intensities in the image. Once these radiomic features are extracted, they are analyzed using machine learning (ML) or advanced statistical methods to identify patterns related to cognitive decline.

### MRI Data Characteristics

- Each MRI scan includes 256 slices (2D images) that together form a 3D image of the brain.
- Each voxel (3D equivalent of a pixel) has an isotropic resolution of 1 mm³, meaning all sides of the voxel measure 1 mm, ensuring uniform spatial accuracy.
- The data is stored in the NIfTI format (Neuroimaging Informatics Technology Initiative), which is commonly used for volumetric neuroimaging data.

You can think of the MRI volume as a 3D image composed of stacked layers, where each voxel represents a 1 mm³ block of brain tissue.
