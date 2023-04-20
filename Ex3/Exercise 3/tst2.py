import cv2
import time
import os
import matplotlib.pyplot as plt

# xmin, xmax, ymin, ymax, class_id
xmin=[110,133,236,244,245,254,255,256,264,385,388]
xmax=[128,141,244,251,251,261,264,263,280,406,405]
ymin=[142,141,123,127,131,126,129,138,140,137,138]
ymax=[153,152,134,138,137,135,137,150,153,154,153]
classid=[1,1,5,5,5,5,5,1,1,1,1]

#%%
root='/home/tuomas/Python/DATA.ML.300/Ex3/Exercise 3/'
img = cv2.imread(root+"1478020228190773357.jpg") 

#%%
for i in range(len(xmin)):
  sp=(xmin[i],xmax[i])
  ep=(ymin[i],ymax[i])
  col=(0,255,0) if classid[i]==1 else (0,0,255)
  th=1
  if classid[i]==5:
      img=cv2.rectangle(img, sp, ep, col, th)
  
cv2.imshow("tst", img)
cv2.waitKey(0)

#%%
i=3
sp=(xmin[i],xmax[i])
ep=(ymin[i],ymax[i])
col=(0,255,0)
th=1

img=cv2.rectangle(img, sp, ep, col, th)
cv2.imshow("tst", img)
cv2.waitKey(0)
#plt.show()
#%%

      


