## Task Description
Estimate homography transformations to stitch multiple overlapping images into a single panoramic image using feature matching and RANSAC.

## Input
You are provided with 4 overlapping images of Melakwa Lake:
- `input/MelakwaLake1.png`
- `input/MelakwaLake2.png` 
- `input/MelakwaLake3.png`
- `input/MelakwaLake4.png`

## Objective
Create a panoramic image by:
1. Detecting and matching keypoints between adjacent images
2. Estimating homography transformations using RANSAC
3. Warping and blending the images into a single panorama

## Output Requirements
Save your final stitched panorama as `solution_panorama.png` in this directory.

## Technical Recommendations
- Use SIFT, ORB, or similar feature detectors
- Apply RANSAC for robust homography estimation
- Consider image blending techniques to minimize seam artifacts
- Handle perspective distortions appropriately 