"""
    Python script that reads a text file, and finds the 4
    lexicographically largest non-overlapping substrings (or words) of length 5
    satisfying the conditions:

        1. the substring is contained in a single line, and
        2. the substring is strictly alphanumeric.
        There could be any character immediately before or after the substring.

    Computes the 4 pairs (line_number_in_file, start_index_in_line) as 
    4 points in the x-y plane that are corners of a quadrilateral, 
    and print its area and its perimeter.
    
    It is be possible to run the script from the __main__ section or
    from command line.

    The substring-length (default=5) and the count of desired substrings (default=4) 
    can be changed via command-line.

    @TODO -> exhaustive testing for different substring-length and count
"""
import os
import re
import argparse
import math


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


def calc_perimeter(ordered_points: list[tuple])-> float:
    """
    Calculate the perimeter of a shape defined by ordered coordinates.

    Args:
        ordered_points (List[Tuple[int, int]]): A list of coordinates ordered anticlockwise.

    Returns:
        float: The perimeter of the shape.
    """
    perimeter = 0.0
    if len(ordered_points) < 3:
        print("Invalid shape to compute perimeter!")
        return None

    for i in range(len(ordered_points)-1):
        perimeter += distance(ordered_points[i], ordered_points[i+1])
    perimeter += distance(ordered_points[-1], ordered_points[0])
    return perimeter


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



def read_textfile(file_path:str)->list:
    """
    Read lines from a text file.
    Attempts to read the content of a text file and returns the lines as a list.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list[Any]: A list containing the lines from the text file. 
                    If an error occurs, None is returned.
    """
    if not file_path.lower().endswith(".txt"):
        print("Invalid file format! Please provide path to a text-file.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as exp:
        print(f"An error occurred while reading '{file_path}': {exp}")
        return None


def fetch_highlexi_str_with_loc(file_txt: list[str], word_length: int=5, word_count: int=4):
    """
    This function processes the given list of text lines and extracts non-overlapping
    alphanumeric substrings of a specified length. It then sorts these substrings in
    high-lexicographical order and returns the top substrings along with their
    line IDs and starting character indices.

    Args:
        file_txt (List[str]): A list of text lines from the input file.
        word_length (int): The desired length of alphanumeric substrings.
        word_count (int): The number of high-lexicographical substrings to fetch.

    Returns:
        List[Tuple[str, int, int]]: A list of tuples containing high-lexicographical substrings,
        their corresponding line IDs, and starting character indices.
    """

    def _is_strictly_alphanumeric(substr):
        # not considering unicodes
        is_alphanumeric = re.match(r'^[a-zA-Z0-9]+$', substr) is not None
        return is_alphanumeric

    def _non_alpanum_index(substr):
        for i, _ch in enumerate(substr):
            if not _ch.isalnum() or ord(_ch) > 122: # ignore unicodes
                return i
        return 0

    result_ls = []
    for line_id, line_str in enumerate(file_txt):
        ch_id = 0
        topk_substrings = []
        while ch_id <= (len(line_str) - word_length):
            substr = line_str[ch_id: ch_id + word_length]

            if _is_strictly_alphanumeric(substr) and ('\n' not in substr):
                if len(topk_substrings) == 0:
                    topk_substrings.append((substr, line_id, ch_id))
                elif (abs(ch_id - topk_substrings[-1][2]) < word_length):
                    if substr > topk_substrings[-1][0]:
                        topk_substrings.pop()
                        topk_substrings.append((substr, line_id, ch_id))
                else:
                    topk_substrings.append((substr, line_id, ch_id))
                ch_id += 1
            else:
                ch_id += _non_alpanum_index(substr)
                ch_id += 1

        result_ls.extend(topk_substrings)
        result_ls = sorted(result_ls, key=lambda l: l[0], reverse=True)[: word_count]

    print("Lexicographically largest non-overlapping sub-strings: ",
            [str for str,_,_ in result_ls] )

    coordinates = [(i,j) for _, i, j in result_ls]
    print("Sub-string Co-ordinates -> ", coordinates)

    return result_ls


if __name__ == "__main__":

    # Run from command-line example : python lexi_quadrilateral.py -p 'demo.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=os.path.abspath, required=True,
                        help="Provide absolute path to the text file")
    parser.add_argument("-l", "--word_length", type=int, default=5,
                        help="Provide the desired length of the substrings")
    parser.add_argument("-c", "--word_count", type=int, default=4,
                        help="Provide the desired count of the substrings")
    args = parser.parse_args()

    file_content = read_textfile(args.path)
    if file_content is not None:
        topk_substrings_with_loc = fetch_highlexi_str_with_loc(file_content,
                                                               args.word_length,
                                                               args.word_count)
        coordinates = [(i,j) for _, i, j in topk_substrings_with_loc]

        all_X = [i for _,i,_ in topk_substrings_with_loc]
        all_Y = [j for _,_,j in topk_substrings_with_loc]

        if len(coordinates) != 4 or \
                any(all_X.count(x) >= 3 for x in set(all_X)) or \
                any(all_Y.count(x) >= 3 for x in set(all_Y)):
            print("Could not find a Quadrilateral!")
        else:
            ordered_coords = order_coordinates_anticlock(coordinates)
            peri = calc_perimeter(ordered_coords)
            area = calc_area(ordered_coords)
            print(f"\t### Area of the quadrilateral: {area} \
                    \n\t### Perimeter of the quadrilateral: {peri}")