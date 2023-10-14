
# UACI(Unified Average Changing Intensity) determines the average change in pixel intensities in corresponding positions in
# the two cipher images C1 and C2 as a percentage with maximum pixel intensity F
# (F = 255 in gray scale images for 8-bit images)
def uaci(img1, img2):
    height, width=img1.shape
    value=0
    for y in range(height):
        for x in range(width):
            value+=(abs(int(img1[x,y])-int(img2[x,y])))
    value=value*100/(width*height*255)
    return value

