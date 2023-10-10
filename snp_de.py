import numpy as np

def logistic_map(x,r):
    sub_array = np.zeros(256,dtype=int)
    for i in range(256):
        x = r * x * (1 - x)
        sub_array[i] = (x*255) % 256
    return sub_array

def inverse(arr):
    temp = np.zeros(len(arr),dtype=int)
    for i in arr:
        temp[arr[i]] = i
    return temp

def substitution_array_generation():
    sub_array = logistic_map(0.01,3.95)
    check = [False] * 256
    least_free_val = 0
    for i in range(256):
        if check[sub_array[i]] == False:
            check[sub_array[i]] = True
        else:
            sub_array[i] = least_free_val
            check[least_free_val] = True
            while(check[least_free_val]):
                least_free_val = least_free_val + 1
    sub_array = inverse(sub_array)
    return sub_array


def row_permutation(image_array,sub_array):
    for i in range(len(image_array)):
        temp = np.zeros(i,dtype=int)
        for k in range(i):
            temp[k] = sub_array[image_array[i][-k-1]]
        for j in range(len(image_array[0])-i-1,-1,-1):
            image_array[i][j+i] = sub_array[image_array[i][j]]
        m = len(temp)- 1
        for l in range(0,i):
            image_array[i][l] = temp[m]
            m = m-1


def col_permutation(image_array,sub_array):
    for i in range(len(image_array[0])):
        temp = np.zeros(i,dtype=int)
        for k in range(i):
            temp[k] = sub_array[image_array[-k-1][i]]
        for j in range(len(image_array)-i-1,-1,-1):
            image_array[j+i][i] = sub_array[image_array[j][i]]
        m = len(temp) - 1
        for l in range(0,i):
            image_array[l][i] = temp[m]
            m = m-1

def snp_decryption(image_array,N):
    sub_array = substitution_array_generation()
    print(sub_array)
    for i in range(N):
        col_permutation(image_array,sub_array)
        row_permutation(image_array,sub_array)
    print(image_array)

test = [[104, 129, 217], [90, 5, 243], [13, 2, 56]]

snp_decryption(test,1)