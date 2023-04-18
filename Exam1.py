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

cv2.imshow("img", gray_img)
plt.hist(matrix)
plt.show()

X1 = np.array([[0,1,0],[1,1,1],[0,1,0]])

X2 = np.array([[1,1,1],[1,1,1],[1,1,1]])

X3 = np.array([[1,3,1],[3,16,3],[1,3,1]])

X4 = np.array([[0,1,0],[1,4,1],[0,1,0]])

X1 = X3 / 32

for i in range(1, matrix.shape[0]-1):
    for j in range(1, matrix.shape[1]-1):
        temp = matrix[i-1, j-1]*X1[0, 0]+matrix[i-1, j]*X1[0, 1]+matrix[i-1, j + 1]*X1[0, 2]+matrix[i, j-1]*X1[1, 0] + matrix[i, j]*X1[1, 1]+matrix[i, j + 1]*X1[1, 2]+matrix[i + 1, j-1]*X1[2, 0]+matrix[i + 1, j]*X1[2, 1]+matrix[i + 1, j + 1]*X1[2, 2]
        matrix_new[i][j] = temp

matrix_new = matrix_new.astype(np.uint8)
cv2.imshow('new_img.jpg', matrix_new)
plt.hist(matrix_new)
plt.show()