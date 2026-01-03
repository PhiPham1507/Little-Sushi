# Eigen-Sushi: PCA Dimensionality Reduction

A mathematical and computational exploration of **Principal Component Analysis (PCA)**. This project implements lossy image compression by projecting high-dimensional image data into a lower-dimensional space while maximizing variance retention.


## Key Features
* Includes a proof (see `PCA_Report.pdf`) demonstrating why eigenvectors of the covariance matrix correspond to the directions of maximum variance.
* Evaluates the trade-off between the number of principal components ($k$) and image quality.
* Compressed an image of a sushi plate from 3,171kb down to 550kb. 
* Analyzes the runtime complexity and numerical stability of $O(m^2n + m^3)$ vs $O(nmk)$ approaches.


## Stack
* **Language:** Python
* **Libraries:** NumPy, OpenCV, Scikit-Learn, Matplotlib
* **Documentation:** LaTeX

## Project Structure
* `main.py`: Core pipeline for compression and reconstruction.
* `PCA_Report.pdf`: Full technical report including proofs and complexity analysis.
* `images/`: Original, reduced, and reconstructed sushi images.
