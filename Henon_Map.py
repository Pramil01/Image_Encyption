import numpy as np

# A Henon map of size n
def Henon_Map(x,y,n,a=1.4,b=0.3):
    x_list = np.zeros(n)
    y_list = np.zeros(n)
    for i in range(n):
        x = 1 - a*x*x + y
        y = b*x
        x_list[i] = int((x + 1) * 127.5) % 256
        y_list[i] = int((y + 1) * 127.5) % 256
    return x_list,y_list

# creating a mask of n * n by xor of x_list and y_list elements
def create_mask(x_list,y_list,n):
    mask  = np.zeros((n,n),dtype=int)
    for i in range(n):
        for j in range(n):
            mask[i][j] = int(x_list[i]) ^ int(y_list[j])
    return mask 

# A function to test the distance between two masks created with diff. initial x and y values
def mask_diff(mask1,mask2):
    n = len(mask1)
    diff = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            diff[i][j] = abs(mask1[i][j] - mask2[i][j])
    return diff

def export_mask(x,y,n):
    x_list,y_list = Henon_Map(x,y,n)
    mask =  create_mask(x_list,y_list,n)
    return mask

def test():
    n = 128
    x,y = Henon_Map(0.001,0.2,n)
    xl,yl = Henon_Map(0.001,0.26,n)
    

    # print(x)
    # print(xl)
    # print(y)
    # print(yl)

    mask =  create_mask(x,y,n)
    maskl = create_mask(xl,yl,n)

    

    # print(mask)
    # print(maskl)

    # print(mask_diff(mask,maskl))

