import matplotlib.pyplot as plt
import numpy as np

def displayData(X, example_width = None):
    '''????'''
    if example_width == None:
        example_width = (int)(np.round(np.sqrt(X.shape[1])))
    m, n = X.shape
    example_height = (int)(n/example_width)

    display_rows = (int)(np.floor(np.sqrt(m)))
    display_cols = (int)(np.ceil(m/display_rows))

    pad = 1
    display_array = - np.ones((pad + display_rows*(example_height+pad),
                               pad + display_cols*(example_width+pad)))
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >=m:
                break
            max_val = np.max(np.abs(X[curr_ex]))
            offset_height = pad + j *(example_height+pad)
            offset_width = pad + j *(example_width+pad)

            display_array[
            offset_height:offset_height+example_height,
            offset_width: offset_width+example_width]\
                =\
                X[curr_ex].reshape((example_height,example_width)).T/max_val

            curr_ex +=1
        if curr_ex >= m :
            break
    plt.imshow(display_array, cmap='gray',vmin=-1,vmax=1)
    plt.axis('off')
    plt.show()
