# preprocess description

There are two main function for preprocess, allocate wire position and cleaning.

## Allocate wire positions
- load each wire position from csv file
- use the middle point of Z axis x =  (x_west+x_east)/2 , y =  (y_west+y_east)/2 
- cal (r, phi) form  (x, y)
- allocate wire pos to each hit

## cleanning 
- cleaning hit
- cleaning track