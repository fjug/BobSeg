from spimagine import volshow
import numpy as np

# create a 3d or 4d numpy array
data = np.linspace(0,1,100**3).reshape((100,)*3)          

# render the data and returns the widget 
w = volshow(data)       

# manipulate the render states, e.g. rotation and colormap
w.transform.setRotation(.1,1,0,1)
w.set_colormap("hot")

# # save the current view to a file
# w.saveFrame("scene.png")