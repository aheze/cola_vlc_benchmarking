Detect all the coins in the image in this directory by: 
1. Using a Canny Edge Detector 
2. Hough Transform

Output your final result in `solution.json` in this directory. 
Its structure should be a list of detections of all coins, for example:

```
{
    "coins": [
        {"x": 55, "y": 50, "radius": 10},
        {"x": 120, "y": 80, "radius": 15}
    ]
}
```