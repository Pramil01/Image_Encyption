import numpy as np

def logistic_map(x,r):
    sub_array = np.zeros(256,dtype=int)
    for i in range(256):
        x = r * x * (1 - x)
        sub_array[i] = int(x * 255)
    return sub_array

def substitution_array_generation():
    sub_array = logistic_map(0.01,3.95)
    check = [False] * 256
    most_free_val = 255
    for i in range(256):
        if check[sub_array[i]] == False:
            check[sub_array[i]] = True
            while(most_free_val != -1 and check[most_free_val] == True):
                most_free_val = most_free_val - 1
        else:
            sub_array[i] = most_free_val
            check[most_free_val] = True
            while(most_free_val != -1 and check[most_free_val] == True):
                most_free_val = most_free_val - 1
    return sub_array

def row_permutation(image_array,sub_array):
    for i in range(len(image_array)):
        temp = np.zeros(i,dtype=int)
        for k in range(i):
            temp[k] = sub_array[image_array[i][k]]
        for j in range(i,len(image_array[0])):
            image_array[i][j-i] = sub_array[image_array[i][j]]
        m = 0
        for l in range(len(image_array[0])-i,len(image_array[0])):
            image_array[i][l] = temp[m]
            m = m+1


def col_permutation(image_array,sub_array):
    for i in range(len(image_array[0])):
        temp = np.zeros(i,dtype=int)
        for k in range(i):
            temp[k] = sub_array[image_array[k][i]]
        for j in range(i,len(image_array)):
            image_array[j-i][i] = sub_array[image_array[j][i]]
        m = 0
        for l in range(len(image_array)-i,len(image_array)):
            image_array[l][i] = temp[m]
            m = m+1

def snp_encryption(image_array,N):
    sub_array = substitution_array_generation()
    for i in range(N):
        row_permutation(image_array,sub_array)
        col_permutation(image_array,sub_array)

