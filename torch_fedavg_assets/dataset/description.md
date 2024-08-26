# VinBigData Chest X-ray Dataset

This dataset comprises chest X-ray images for the detection and classification of common thoracic lung diseases. It is part of the VinBigData Chest X-ray Abnormalities Detection competition.

## Dataset Details

- **Type**: Postero-anterior (PA) CXR scans
- **Format**: DICOM
- **Number of Classes**: 14 critical radiographic findings + 1 "No finding" class
- **Image Size**: 224x224 pixels (resized from original)

## Classes

0. Aortic enlargement
1. Atelectasis
2. Calcification
3. Cardiomegaly
4. Consolidation
5. ILD
6. Infiltration
7. Lung Opacity
8. Nodule/Mass
9. Other lesion
10. Pleural effusion
11. Pleural thickening
12. Pneumothorax
13. Pulmonary fibrosis
14. No finding

This dataset has been preprocessed and split into training and testing sets for each organization in the federated learning setup.