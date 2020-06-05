import numpy as np
from PIL import Image
import cv2
import sys

'''
Sections of the work have been adapted from:
https://github.com/andrewdcampbell/seam-carving/blob/master/seam_carving.py
'''

MAX_WIDTH = 500

def get_forward_energy(image):
    '''
    Implement forward energy function to determine optimal place to do seams
    '''
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    height = image.shape[0]
    width = image.shape[1]

    energy = np.zeros((height, width))
    m = np.zeros((height, width))

    upper = np.roll(image, 1, axis = 0)
    left = np.roll(image, 1, axis = 1)
    right = np.roll(image, -1, axis = 1)

    # Get cost of connecting top seam, left seam, and right seam
    cost_upper = np.abs(right - left)
    cost_left = np.abs(upper - left) + cost_upper
    cost_right = np.abs(upper - right) + cost_upper

    for i in range(1, height):
        m_upper = m[i - 1]
        m_left = np.roll(m_upper, 1)
        m_right = np.roll(m_upper, -1)

        m_all = np.array([m_upper, m_left, m_right])
        cost_all = np.array([cost_upper[i], cost_left[i], cost_right[i]])

        m_all += cost_all

        #Get index of minimum energy
        min_idx = np.argmin(m_all, axis = 0)
        m[i] = np.choose(min_idx, m_all)

        #Select minimum energy from costs
        energy[i] = np.choose(min_idx, cost_all)

    return energy


def min_seam(image):
    '''
    Get the seam with the minimum energy in the image. Uses forward energy
    @param image: the input image

    Code adapted from: 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    '''
    height = image.shape[0]
    width = image.shape[1]

    # Get energies from forward energy function
    energy = get_forward_energy(image) 
 
    # Array to hold values of minimum seam to backtrack through to remove
    seam = np.zeros_like(energy, dtype = np.int) 

    for i in range(1, height):
        for j in range(0, width):
            # Handle leftmost column to stay in bounds of image
            if j == 0: 
                min_idx = np.argmin(energy[i - 1, j: j + 2])
                seam[i, j] = min_idx + j
                min_energy = energy[i - 1, min_idx + j]
            else:
                min_idx = np.argmin(energy[i - 1, j - 1:j + 2])
                seam[i, j] = min_idx + j - 1
                min_energy = energy[i - 1, min_idx + j - 1]

            energy[i, j] += min_energy

    return energy, seam


def backtrack(image, energy, seam):
    '''
    Start at the bottom of the image and work backwards, getting the index of the seam
    with respect to the distance from the left hand side
    @param image: the image to get the seam from
    @param energy: array containing the energy functions of the image
    @param seam: the array containing the seam calculated by min_seam()
    '''
    height = image.shape[0]
    width = image.shape[1]

    seam_col = []
    mask = np.ones((height, width), dtype = np.bool)
    j = np.argmin(energy[-1])

    # Start at the bottom of the image and walk back up to the top following
    # the path of the seam 
    for i in range(height - 1, -1, -1):
        mask[i, j] = False
        seam_col.append(j)
        j = seam[i, j]

    seam_col.reverse()
    seam_col = np.array(seam_col)

    return seam_col, mask


def downsize(im, height, width):
    '''
    Downsizes an image to be of width MAX_WIDTH and scales the image such that
    the image stays in its original proportions
    @param im: the input image, with original dimensions
    @param height: height of original image
    @param width: width of original image
    '''
    dimension = (MAX_WIDTH, int(height*MAX_WIDTH/float(width)))

    return cv2.resize(im, dimension)


def remove_seam(image, image_vis, mask, seam):
    '''
    Removes a seam from the image. Projects the boolean mask over the three
    channels of the image (for RGB). 
    @param image: input image to remove seam from
    @param mask: 1D boolean mask to convert to number of channels in image and remove
    from image
    '''
    result_vis = np.copy(image_vis)
    height, width, channels = image.shape

    # Make mask same number of channels as image
    c_channel_mask = np.stack([mask]*channels, axis=2)

    # Visualize removed seams as white (255), remove seams from image
    result_vis[np.where(c_channel_mask == False)] = 255
    image = image[c_channel_mask].reshape((height, width-1, channels))

    return image, result_vis
    

def remove_seams(image, image_vis, num_seams):
    '''
    Removes the number of seams requested in the input image. Uses min_seam(),
    backtrack(), and remove_seam() as helper functions.
    @param image: input image from which to remove seams 
    @param num_seams: number of seams to remove
    '''
    for seam in range(num_seams):
        energy, seam = min_seam(image)
        seam_col, mask = backtrack(image, energy, seam)
        image, image_vis = remove_seam(image, image_vis, mask, seam)

    return image, image_vis

def add_seam(image, image_vis, seam_col):
    '''
    Add vertical seam to color image at seam index given.
    @param image: input image from which to add seam
    @param image_vis: visualization image that includes lines to show
    added seams
    @seam_col: path of seam to add
    '''

    height, width, num_channels = image.shape

    # Initialize result as array of zeros with 
    # image height, width + 1 for added seam, and 3 color channels
    result = np.zeros((height, width + 1, 3))
    result_vis = np.zeros((height, width + 1, 3))

    # Initialize array to store red for visualization lines
    red_array = [255, 0, 0]

    for i in range(1, height):
        column = seam_col[i]
        for channel in range(num_channels):
            if column == 0: #Handle column 0
                pixel_color = np.average(image[i, column: column + 2, channel]) #Average pixels from image at column and column + 2
                result[i, column, channel] = image[i, column, channel]
                result[i, column + 1, channel] = pixel_color #Set color at column + 1 to pixel_color
                result[i, column + 1:, channel] = image[i, column:, channel]
                # Do the same process for visualization, except adding red
                result_vis[i, column, channel] = image_vis[i, column, channel]
                result_vis[i, column + 1, channel] = red_array[channel]
                result_vis[i, column + 1:, channel] = image_vis[i, column:, channel]
            else:
                pixel_color = np.average(image[i, column - 1: column + 1, channel])
                result[i, : column, channel] = image[i, : column, channel]
                result[i, column, channel] = pixel_color
                result[i, column + 1:, channel] = image[i, column:, channel]
                # Do the same process for visualization, except adding red
                result_vis[i, : column, channel] = image_vis[i, : column, channel]
                result_vis[i, column, channel] = red_array[channel]
                result_vis[i, column + 1:, channel] = image_vis[i, column:, channel]

    return result, result_vis
    

def insert_seams(image, image_vis, num_seams):
    '''
    Add given number of seams to image
    @param image: input image from which to add seam
    @param image_vis: visualization image that includes lines to show
    added seams
    @param num_seams: number of seams to add
    '''
    seam_list = []
    temporary = image.copy()

    for i in range(num_seams):
        #Get minimum seam and mask from temporary image and append to seam_list
        energy, seam = min_seam(temporary)
        seam_col, mask = backtrack(temporary, energy, seam)
        seam_list.append(seam_col)
        #Remove seam from temporary image
        temporary, temp_vis = remove_seam(temporary, image_vis, mask, 0)

    #Reverse seam list
    seam_list.reverse()

    for j in range(num_seams):
        #Get minimum seam from seam_list and add to image
        seam_to_add = seam_list.pop()
        image, image_vis = add_seam(image, image_vis, seam_to_add)

        #Update other seams to consider added seam
        for seam in seam_list:
            seam[np.where(seam >= seam_to_add)] += 2   

    return image, image_vis

def run(image_name, v_seams=0, h_seams=0):
    '''
    Main driver function
    '''
    im = np.array(Image.open(image_name))
    height, width = im.shape[0], im.shape[1]

    # Downsize the image for added speed
    if (width > MAX_WIDTH):
        im = downsize(im, height, width)

    # If no seams are given, the original image is saved and returned
    if (h_seams == 0) and (v_seams == 0):
        im = Image.fromarray(im) 
        final_image_name = image_name.split("/static/", 1)[1]
        final_image_name = final_image_name.split(".", 1)
        final_image_name = "static/" + final_image_name[0] + \
                            "_carved." + final_image_name[1]
        im.save(final_image_name)
        print("No seams were provided")
        return final_image_name, final_image_name
    # Make sure seam remove isn't greater than number of pixels
    elif (h_seams + height <= 0) or (v_seams + width <= 0):
        im = Image.fromarray(im)
        final_image_name = image_name.split("/static/", 1)[1]
        final_image_name = final_image_name.split(".", 1)
        final_image_name = "static/" + final_image_name[0] + \
                            "_carved." + final_image_name[1]
        im.save(final_image_name)
        print("Cannot remove more seams than are in picture")
        return final_image_name, final_image_name
    
    final_image = im.astype(np.float64)
    image_vis = final_image
    
    # First do the horizontal seams, and call either remove or insert depending on sign
    if h_seams < 0:
        final_image, image_vis = remove_seams(final_image, image_vis, -h_seams)
    elif h_seams > 0:
        final_image, image_vis = insert_seams(final_image, image_vis, h_seams)
    
    # Then rotate the image and do the vertical seams
    if v_seams < 0:
        # Rotate clockwise to do the removal
        final_image = np.rot90(final_image, k=1)
        image_vis = np.rot90(image_vis, k=1)
        final_image, image_vis = remove_seams(final_image, image_vis, -v_seams)
        # Rotate back to correct orientation
        final_image = np.rot90(final_image, k=3)
        image_vis = np.rot90(image_vis, k=3)

    elif v_seams > 0:
        # Rotate clockwise to do the removal
        final_image = np.rot90(final_image, k=1)
        image_vis = np.rot90(image_vis, k=1)
        final_image, image_vis = insert_seams(final_image, image_vis, v_seams)
        # Rotate back to correct orientation
        final_image = np.rot90(final_image, k=3)
        image_vis = np.rot90(image_vis, k=3)
    
    # Get the names for the two final files
    final_image_name = image_name.split("/static/", 1)[1]
    final_image_name = final_image_name.split(".", 1)
    final_image_vis_name = "static/" + final_image_name[0] + \
                        "_vis." + final_image_name[1]
    final_image_name = "static/" + final_image_name[0] + \
                        "_carved." + final_image_name[1]

    final_image = final_image.astype(np.uint8)
    final_image_vis = image_vis.astype(np.uint8)

    # Save the final image
    final_image = Image.fromarray(final_image)
    final_image.save(final_image_name)

    # Save the visualization
    final_image_vis = Image.fromarray(final_image_vis)
    final_image_vis.save(final_image_vis_name)

    return final_image_name, final_image_vis_name