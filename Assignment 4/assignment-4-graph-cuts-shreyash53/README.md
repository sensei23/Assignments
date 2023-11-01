[![Generic badge](https://img.shields.io/badge/CV-Assignment:4-BLUE.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/DUE-23:59hrs,14/03/2023-RED.svg)](https://shields.io/)

# Assignment-04

The goal of the assignment is to familiarize you with graph cuts and their application in computer vision problems.

Please raise doubts, if any on the appropriate assignment thread on Moodle.

<br>

# Instructions

-   Follow the directory structure as shown below:
    ```
    ├── src
          ├── Assignment04.ipynb
    ├── data       // Given input images
    ├── results    // Store your results here
    ├── Assign04.pdf
    └── README.md
    ```
-   `src` will contain the Jupyter notebook(s)/ python file(s) used for the assignment.
-   You are supposed to write all of your code in python.
-   **Make sure you run your Jupyter notebook before committing, to save all outputs.**

<br>

# Implementation Details

## Disparity maps in stereo images

-   You can assume that images are epipolar rectified and there is purely horizontal motion between the two stereo images.
-   You can either hardcode values of $\lambda$ and the occlusion cost ($K$), or can calculate them as described in section 4.2 of IPOL article.
-   You are given a helper function to convert ground truth disparity image into a disparity map. Feel free to edit it as you seem fit.

-   While calculating the evaluation metrics, you need to crop the disparity maps. You need to remove 18 columns from the left and right side of the map, and similarly 18 rows from the top and the bottom. This is because the ground truth disparity image hasn't reported the disparity values for this region. You can see the helper function for this as well.

-   Helper numpy functions
    -   `numpy.zeros`, `numpy.ones`, `numpy.full`, `numpy.arange`
    -   `numpy.max`, `numpy.mean`, `numpy.min`, `numpy.sum`
    -   `numpy.indices`, `numpy.ndindex`
    -   `numpy.concatenate`, `numpy.hstack`, `numpy.vstack`, `numpy.column_stack`
    -   `numpy.clip`
    -   `numpy.where`
    -   `numpy.vectorize`
