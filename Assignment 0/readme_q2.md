# Problem
1. Given a video. Save all video frames as images to a folder.  
2. Given a folder of images, combine the images to form a video.
<br>

# Solution
1. we use the video capture method in the library to load the video, and then while reading all frames simply store the frames in image folder.
2. we use the video writer method of the library to represent what video is to be created, with a codec of 'DIVX' along with frame rate and frame size. Now we just load the images one by one and write them into the video.

<br>

# Challenges
the video writing codec was something new. I initially when tried to store the video with an extension of .mp4, and tried using the fourcc codec with 'DIVX' format, then the video was not displaying. Then by changing the extenshion to .avi, the issue was resolved.


<br>

# Learnings
- videocapture and videowriter in the python opencv library.
- use of codecs to write an image
- converting an video to images and vice-versa
