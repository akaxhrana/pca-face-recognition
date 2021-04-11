# pca-face-recognition
Face recognition using PCA and MLPClassifier

## Usage
  - Clone the repo and install through Pipfile.
  - ```mkdir data``` and put photos of same person in a directory named after that person.
  - ```python main.py``` will
      - Extract faces from the images
      - Convert faces extracted into numpy arrays and reshape for dataset
      - Reduce the dimensions present in dataset (to speed up training model without a loss of variance)
      - Fit the reduced dimensions of data into MLPClassifier
      - Show classification report
  - Use "_main.ipynb_" as per your wish
