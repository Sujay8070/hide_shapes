"""
A Python script that reads an image as grayscale,
and finds the four non-overlapping 5x5 patches with highest average
brightness. Takes the patch centers as corners of a quadrilateral,
calculates its area in pixels, and draws the quadrilateral in red into
the image and saves it in PNG format.

It is be possible to run the script from the __main__ section or
from command line.

The patch_size (default=5x5) and the count of patches (default=4) can 
also be changed via command-line.

@TODO -> exhaustive testing for another patch size and patch counts


"""
import os
import argparse

import cv2
import math
import numpy as np


def read_image(im_path:str) -> tuple:
    """
    Read and process an image from the specified path.
    
    Args:
        im_path (str): Path to the image file.
        
    Returns:
        tuple: A tuple containing the original color image (BGR format) and its grayscale version.
    """

    try:
        orig_img = cv2.imread(im_path)
        if orig_img is None:
            raise ValueError("Error: Unable to read the image. Please check the image file format")
        
        if orig_img.shape[0] < 5 or orig_img.shape[1] < 5:
            print("Insufficient image size!")
            return None, None

        gray_im = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) 
    except Exception as exp:
        print(f"An error occurred: {exp}")
        if not im_path.lower().endswith(".png") or \
            not im_path.lower().endswith(".jpg"):
            print("Please provide the image in 'jpg' or 'png' file format.")
        return None, None

    return orig_img, gray_im


def get_allpatches_with_brightness_value(gray_im: np.ndarray,
                                         patch_size: int=5,
                                         max_patch_count:int=4) -> list[int,int]:
    """
    Extract and rank image patches based on their average brightness value. 
    Returns the center locations of selected non-overlapping
    patches (that have maximum average brightness).

    Args:
        gray_im (np.ndarray): Grayscale image in the form of a NumPy array.
        patch_size (int, optional): Size of the square patches to be extracted. Default is 5.
        max_patch_count (int, optional): Maximum number of patches to be extracted. Default is 4.
        
    Returns:
        list: List of center points (as [row, column] pairs) of the top ranked patches based on brightness.
    """
    if gray_im.shape[0] < 5 or gray_im.shape[1] < 5:
        return None

    def _check_overlap(test_patch, patch_list, patch_size):
        if len(patch_list) == 0:
            return False
        for _patch in patch_list:
            if (abs(test_patch[1] - _patch[1]) < patch_size) and \
                    (abs(test_patch[2] - _patch[2]) < patch_size):
                return True
        return False

    total_patches = []
    for row in range(gray_im.shape[0] - patch_size + 1):
        for col in range(gray_im.shape[1] - patch_size + 1):
            curr_patch = gray_im[row : row + patch_size, col: col + patch_size]
            patch_brightness = np.mean(curr_patch)
            total_patches.append((patch_brightness, row, col))
    total_patches = sorted(total_patches, key=lambda l: l[0], reverse=True)

    topk_patches = []
    for patch in total_patches:
        if len(topk_patches) < max_patch_count:
            if not _check_overlap(patch, topk_patches, patch_size):
                topk_patches.append(patch)
        else:
            break
    centers = [[i + patch_size//2, j + patch_size//2] for _,i,j in topk_patches]

    print(f"Center points of the brightest patches: {centers}")

    return centers


def distance(p1:tuple[int, int],
             p2:tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.
    """
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def calc_polar_angle(p1: tuple[int, int],
                     p2: tuple[int, int]) -> float:
    """
    Calculate the polar angle between two points.

    Args:
        p1 (Tuple[int, int]): The coordinates of the first point (x1, y1).
        p2 (Tuple[int, int]): The coordinates of the second point (x2, y2).

    Returns:
        float: The polar angle between the two points in radians.
    """
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def order_coordinates_anticlock(points: list[tuple])-> list[tuple]:
    """
    Order a list of coordinates in an anticlockwise sequence.

    Args:
        points (List[Tuple[int, int]]): A list of tuples representing (x, y) coordinates.

    Returns:
        List[Tuple[int, int]]: A list of coordinates sorted in an anticlockwise order.
    """
    if len(points) == 0:
        return []
    centroid = [sum(x for x,_ in points) / len(points), sum(y for _, y in points) / len(points)]
    points.sort(key=lambda p: (calc_polar_angle(centroid, p), - distance(centroid, p)))
    return points



def calc_area(ordered_points: list[tuple])-> float:
    """
    Computes the area of the quadrilateral, given 4 sets of coordinates
    Function taken from : https://www.geodose.com/2021/09/how-calculate-polygon-area-unordered-coordinates-points-python.html
       
    Args:
        ordered_points (_type_): List of 4 tuples, each as (x, y) integer co-ordinates

    Returns:
        float: area enclosed by the 4 points
    """

    if len(ordered_points) < 3:
        print("Invalid shape to compute area!")
        return None

    # separate X AND Y lists
    def _separate_xy(xy_pairs:list[tuple] ) -> tuple[list]:
        x_ls=[]
        y_ls=[]
        for _,_pt in enumerate(xy_pairs):
            x_ls.append(_pt[0])
            y_ls.append(_pt[1])
        return x_ls, y_ls

    # area of regular-polygon, from ordered coordinates
    def _shoelace_area(x_list, y_list):
        a1, a2 = 0, 0
        x_list.append(x_list[0])
        y_list.append(y_list[0])
        for j in range(len(x_list)-1):
            a1 += x_list[j]*y_list[j+1]
            a2 += y_list[j]*x_list[j+1]
        Area = abs(a1-a2)/2
        return Area

    xy_e = _separate_xy(ordered_points)
    return _shoelace_area(xy_e[0], xy_e[1])


# Draw red quadrilateral box
def draw_quadrilateral(img, points: list[tuple], save_at: str="output_image.png"):
    """
    Draw a quadrilateral on an image using given points and save the result.

    Args:
        img (numpy.ndarray): The input image.
        points (list of tuples): List of 4 tuples representing (x, y) coordinates of the quadrilateral vertices.
        save_at (str, optional): Path to save the output image. Default is "output_image.png".

    Returns:
        Boolean: whether quadrilateral is drawn on the image and saved successfully.
    
    """

    all_X = [i for i,_ in points]
    all_Y = [j for _,j in points]
    if len(points) != 4 or \
        any(all_X.count(x) >= 3 for x in set(all_X)) or \
        any(all_Y.count(x) >= 3 for x in set(all_Y)):
        print("Could not find a Quadrilateral!")
        return False
    
    ordered_pts = order_coordinates_anticlock(points)
    print(f"Quadrilateral co-ordinates: {ordered_pts}")
    
    quad_area = calc_area(ordered_points=ordered_pts)
    print(f"Area of the formed Quadrilateral: {quad_area} sq. pixels")
    
    box = [[y,x] for [x,y] in ordered_pts]
    box = np.array(box).reshape((-1, 1, 2))
    color = (0, 0, 255) # BGR
    th=1 if img.shape[0] > 300 else 2
    cv2.polylines(img, [box], isClosed=True, color=color, thickness=th)
    
    if cv2.imwrite(save_at, img):
        print(f"New image saved at {save_at}.")
    return True


if __name__ == "__main__":

    ## Console command -> python img_bright_quadrilateral.py -p 'input_im.jpg'

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=os.path.abspath, required=True,
                        help="Provide absolute path to the image file")
    parser.add_argument("-s", "--patch_size", type=int, default=5,
                        help="Provide the desired size of a patch")
    parser.add_argument("-c", "--patch_count", type=int, default=4,
                        help="Provide the total number of patches to be fetched")
    parser.add_argument("-l", "--img_save", type=str, default="output_image.png",
                        help="Provide the location to save the output image")
    args = parser.parse_args()

    original_im, grascale_im = read_image(im_path = args.path)

    if original_im is not None:
        center_points = get_allpatches_with_brightness_value(gray_im = grascale_im,
                                            patch_size=args.patch_size,
                                            max_patch_count= args.patch_count)

        draw_quadrilateral(img=original_im, points=center_points, save_at = args.img_save)
