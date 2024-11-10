Image Quantization with K-Means Clustering
==========================================

This project applies K-Means clustering to perform image quantization, 
reducing the number of colors in an image while preserving its visual appeal. 
It includes a comparison of quantized images at different cluster levels and 
demonstrates how quantization can be used to enhance visual aesthetics.

Features
--------
- K-Means Quantization: Reduces the number of colors in images.
- Elbow Method Visualization: Plots inertia values against cluster sizes to evaluate optimal k.
- Task Comparison: Uses quantization from one image to transform another.

How It Works
------------
1. K-Means Clustering
   - Runs multiple K-Means clustering operations with different values of k.
   - Captures inertia (sum of squared distances) for each k to analyze cluster efficiency.

2. Quantization
   - Replaces pixel values with their nearest cluster center.
   - Demonstrates how different values of k affect visual quality.

3. Task 2 Transformation
   - Applies the optimal quantization model from Task 1 to a new image for aesthetic comparison.

Dependencies
------------
- Python 3.8+
- NumPy
- scikit-image
- scikit-learn
- Matplotlib

Setup
-----
1. Clone the repository:
   git clone https://github.com/JordanJancic/KMeansQuantizer

2. Install dependencies:
   pip install -r requirements.txt

3. Run the script:
   python quantization.py

Conclusion
----------
Quantization improves visual aesthetics with reduced color complexity. 
This project highlights the trade-off between image quality and computational efficiency 
through K-Means clustering.
