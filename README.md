# Chest-X-Ray-Image-Classifier
Utilizing Convolutional Autoencoders and Support Vector Machine (SVM) to preprocess, compress, and classify x-ray images for disease diagnosis. 

## Introduction/Background
Medical imaging is an indispensable component of modern healthcare. In particular, chest X-rays are prominent in diagnosing a spectrum of respiratory diseases, including pneumonia, tuberculosis, and, more recently, COVID-19. Pneumonia remains a global health challenge, with high morbidity and mortality rates [**[1]**](#ref1). The COVID-19 pandemic also underscored the importance of accurate and swift diagnosis of infectious diseased. Machine learning (ML) algorithms provide opportunities to automate diagnoses by detecting patterns and anomalies in medical images. The dataset chosen for this project [**[2]**](#ref2) contains many chest X-ray images labeled to indicate the presence of pneumonia, COVID-19, tuberculosis, or their absence. I originally completed this project with 4 other individuals and have modified our code and deliverables for use in my own portfolio. 

## Methods
### Preprocessing
During preprocessing, my team segregated images into folders containing like-classifications, standardized image dimensions and orientations, converted all images into grayscale, and flattened images into a workable format based on pixel values.

### Autoeconders 
Autoencoders, a prevalent method for compression in medical imaging [**[3]**](#ref3), consists of neural networks that encode input images into a lower-dimensional representation and reconstruct the image from this compressed form, aiming to retain key features while minimizing reconstruction error.

The encoder consists of three blocks that map the initial image to a lower dimensional representation, and the decoder consists of three blocks that map the image back to the original dimensions by reversing the transformations during encoding.

More specifically:
- Each block consists of:
  - A convolutional layer to capture the most relevant features for each image
  - A batch normalization layer to alleviate vanishing gradient issues
  - A Leaky-ReLU activation function to capture non-linearities and help to prevent the optimizer from settling in local solutions
  - A MaxPooling layer to reduce dimensions (in the encoder) and a UpSampling Layer to recover original data dimensions

Some details on training:
- The autoencoder is trained with all images together because some classes have fewer observations than others
- Pixels were normalized to the range [0,1] and model performance was assessed using binary-cross-entropy loss
- We used the Adam optimizer with 0.001 learning rate, and batch sizes of 4, 8, 16 and 32 (all with similar performance)

### Support Vector Machine (SVM)
A widely accepted technique for medical image classification [**[4]**](#ref4), I used a SVM to classify the reconstructed images. My team experimented with utilizing Convolutional Neural Networks (CNN) to classify images; however, SVM was able to outperform CNN in both accuracy and efficiency. 

I used GridSearchCV and C-Support Vector Classification (SVC) with 3 kernel types, multiple degrees for polynomial kernels, and multiple coefficients to determine the best blend of hyperparameters to classify the images. 

Most effective hyperparameters:
- A polynomial kernel to capture smooth differences in data clusters
- A third degree polynomial indicates relatively smooth data divisions
- scaled gamma defines the impact of granularity in the data
- coef0 = 1 indicates the intercept term

## Observed Results

#### Training Data Results
Overall Accuracy: 99.27%
| | COVID-19 | Normal | Pneumonia | Tuberculosis |
| --------- | --------- | --------- | --------- | --------- |
| precision | 0.986813 | 0.993980 | 0.993316 | 0.990798 |
| recall | 0.976087 | 0.985086 | 0.997161 | 0.993846 |
| f-1 score | 0.981421 | 0.989513 | 0.995235 | 0.992320 |

#### Testing Data Results
Overall Accuracy: 99.62%
| | COVID-19 | Normal | Pneumonia | Tuberculosis |
| --------- | --------- | --------- | --------- | --------- |
| precision | 1.000000 | 0.993789 |  0.99802 | 0.985507 |
| recall | 0.984848 | 0.993789 | 0.99802 | 1.000000 |
| f-1 score | 0.992366 | 0.993789 | 0.99802 | 0.992701 |

Per the results, our model produces incredible accuracy and minimizes false negative results as observed through an excellent recall score for Normal labelled images. We also minimized overfitting training data by improving accuracy in testing results. 

## Folders
#### Data
- Original data divided into subfolders of train, test, validation and further divided by label (COVID-19, Normal, Pneumonia, Tuberculosis)
#### Data_processed
- Standardized data reduced to uniform size, retains the same folder structure as ./Data folder
#### Data_compressed
- Compressed and reconstructed data utilizing autoencoder, retains the original folder structure
#### Autoencoder
- Contains files used for image compression and preprocessing utilizing a convolutional autoencoder
#### trained_models
- Contains the autoencoder model created for image compression
#### utils
- Utility scripts used to navigate file paths and conduct image preprocessing

## Files
#### Chest X-Ray Image Classifier
- Performs classification on compressed images and displays results

## References:
<a id="ref1"></a> [1] Faden, H., & El-Sharif, N. (2019). The global and regional prevalence of community-acquired pneumonia in children under five in 2010. InTechOpen.

<a id="ref2"></a> [2] Kaggle Dataset: "Chest X-ray - Pneumonia, COVID-19, Tuberculosis." Retrieved from https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis

<a id="ref3"></a> [3] Maier, A., Syben, C., Lasser, T., & Riess, C. (2019). A gentle introduction to deep learning in medical image processing. Zeitschrift für Medizinische Physik, 29(2), 86-101. 

<a id="ref4"></a> [4] E. Miranda, M. Aryuni and E. Irwansyah, "A survey of medical image classification techniques," 2016 International Conference on Information Management and Technology (ICIMTech), 2016, pp. 56-61.
