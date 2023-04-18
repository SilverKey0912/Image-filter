import numpy as np
import cv2
import matplotlib.pyplot as plt

# C:\Users\IT Supporter\Pictures\pnj1.png
# C:\Users\IT Supporter\Pictures\pnj2.png
# C:\Users\IT Supporter\Pictures\pnj3.png
# C:\Users\IT Supporter\Pictures\pnj4.png
# C:\Users\IT Supporter\Pictures\pnj5.png


img = cv2.imread(r'C:\Users\IT Supporter\Pictures\pnj5.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
matrix = np.array(gray_img)
matrix_new = np.zeros([matrix.shape[0], matrix.shape[1]])

cv2.imshow("old image", gray_img)
plt.hist(matrix)
plt.show()

for i in range(1, matrix.shape[0]-1):
    for j in range(1, matrix.shape[1]-1):
        temp = [matrix[i - 1, j - 1],
                matrix[i - 1, j],
                matrix[i - 1, j + 1],
                matrix[i, j - 1],
                matrix[i, j],
                matrix[i, j + 1],
                matrix[i + 1, j - 1],
                matrix[i + 1, j],
                matrix[i + 1, j + 1]]
        temp = sorted(temp)
        matrix[i, j] = temp[4]

cv2.imshow('new_img.jpg', matrix)
plt.hist(matrix)
plt.show()