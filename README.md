# Simple ORT Detection App

Every x frames upon turning on the camera, the image will be analyzed in search of heads, and if 
the detector model finds one, the image will be classified by head position (center, left, right...), 
and if said position is center, the image will be cropped to show only the left eye, and then it will 
be classified by gaze direction
