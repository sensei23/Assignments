# Problem
how to capture frames from a webcam connected to your computer and save them as images in a folder. Also,
be able to display the frames (the video) on the screen while capturing.

<br>

# Solution
using the videocapture library, we enable the use of the webcam. Then, while we are reading the frames from the webcam, we wait until a small threshold to allow for an keyboard input. If the right key is pressed, we capture and store the frame in local storage.

<br>

# Learnings
- use of webcam in applications
- use of methods like imwrite, destroyallwindows, waitkey etc.