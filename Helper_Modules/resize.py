import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import os
# from resize.squircle_master.squircle import to_square
from Helper_Modules.circle_stretcher import to_square
from PIL import Image

def find_sequence_ranges(lst):
    """gets the starting and ending index for each sequence in the list"""
    ranges = []
    start = lst[0]
    end = lst[0]
    
    for i in range(1, len(lst)):
        if lst[i] == end + 1:
            end = lst[i]
        else:
            ranges.append((start, end))
            start = lst[i]
            end = lst[i]
    
    ranges.append((start, end))
    
    result = []
    for seq in ranges:
        result.append(seq[1]-seq[0])

    return result, ranges


def crop_black_columns(image):
    width, height = image.size
    black_columns = []
    
    pixels = np.asarray(image)
    start = int(height/2 - 400) 
    end = int(height/2 + 400)
    for x in range(width):
        # just check the 400 pixels in the middle, so it does't crop the columns where the black pixels are overpowering up and down
        if np.median(pixels[:,x,:]) < 8: 
            black_columns.append(x)
  
    if black_columns:
        X, ranges = find_sequence_ranges(black_columns)
        # taking the first and last garanties that no columns in the middle of the image will be taken as bounds
        min_x = X[0] 
        max_x = width - X[len(X)-1] if black_columns else width 
        if max_x < min_x:
            cropped_image = image
        else:
            cropped_image = image.crop((min_x, 0, max_x, height))
        return cropped_image
    else:
        return image
    
def to_square_save(image_path, new_path=None):
    """Turns the circular image in a black frame into square,and returns the new imge or saves it in case a new path is given.\n 
    The given path should be the root path of the new dataset.
    The saved image will have the same name, the same parent dir(class), and the same second parent name(split)"""
    image = Image.open(image_path)
    cropped_image = crop_black_columns(image)
    cropped_image = cropped_image.resize([224, 224])
    image_array = np.array(cropped_image)
    square = to_square(image_array)
    im = Image.fromarray(square)
    if new_path == None:
        return im
        
    else:
        dest_directory = os.path.join(new_path, image_path.parts[-3], image_path.parts[-2])
        os.makedirs(dest_directory, exist_ok=True)
        new_path = os.path.join(dest_directory, image_path.name)
        im.save(new_path)
        return new_path