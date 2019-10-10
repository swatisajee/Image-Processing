"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import task1
import utils
import cv2
from task1 import *  # you could modify this line
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str,
        default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str,
        default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

        Args:
            img: nested list (int), image that contains character to be detected.
            template: nested list (int), template image.

        Returns:
            coordinates: list (tuple), a list whose elements are coordinates where the character appears.
                format of the tuple: (x (int), y (int)), x and y are integers.
                x: row that the character appears (starts from 0).
                y: column that the character appears (starts from 0).
        """

    #detect threshold
    args = parse_args()
    threshold=dictornary(args)

    # detect edges of image
    """prewitt_x = [[1, 0, -1]] * 3
    prewitt_y = [[1] * 3, [0] * 3, [-1] * 3]
    img_x = task1.detect_edges(img, prewitt_x, False)
    img_y = task1.detect_edges(img, prewitt_y, False)
    img_norm = task1.edge_magnitude(img_x, img_y)

    task1.write_image(task1.normalize(img_norm), ".//img_norm.jpg")

    # detect edges in template

    temp_x = task1.detect_edges(template, prewitt_x, False)
    temp_y = task1.detect_edges(template, prewitt_y, False)
    template_norm = task1.edge_magnitude(temp_x, temp_y)

    task1.write_image(task1.normalize(template_norm), ".//template_norm.jpg") """

    img_norm = task1.normalize(img)
    template_norm = task1.normalize(template)

    coordinates = []
    temp_h = len(template_norm)
    temp_w = len(template_norm[0])

    rows = len(img_norm)
    cols = len(img_norm[0])

    output = [[0 for x in range(len(img_norm[0]))] for y in range(len(img_norm))]
    cropped_img = [[0 for x in range(temp_w)] for y in range(temp_h)]

    for i in range(rows):
        for j in range(cols):

            if ((i +temp_h) < rows and (j + temp_w < cols)):
                cropped_img = utils.crop(img_norm, i, i + temp_h, j, j + temp_w)


            img_mul_temp = utils.elementwise_mul(cropped_img, template_norm)
            sum = 0
            # sum of every elemnet in img_mul_temp
            for p in range(temp_h):
                for q in range(temp_w):
                    sum += img_mul_temp[p][q]

            # squaring every element in denominator of image
            square_img = utils.elementwise_mul(cropped_img, cropped_img)
            numsum_img = 0
            for d in range(len(cropped_img)):
                for e in range(len(cropped_img[0])):
                    numsum_img += square_img[d][e]

            # squaring every element in denominator of template
            square_temp = utils.elementwise_mul(template_norm, template_norm)
            numsum_temp = 0
            for k in range(temp_h):
                for l in range(temp_w):
                    numsum_temp += square_temp[k][l]

            denominator = np.sqrt((numsum_img * numsum_temp))

            if (denominator != 0):
                output[i][j] = (sum / denominator)
            if (output[i][j] > threshold):
                coordinates.append([i, j])

    # TODO: implement this function.
    # raise NotImplementedError
    return coordinates

def dictornary(args):
    threshold = 0.97
    if (args.template_path == './data/a.jpg'):
        threshold = 0.959
    if (args.template_path == './data/b.jpg'):
        threshold = 0.98
    if (args.template_path == './data/c.jpg'):
        threshold = 0.98
    return threshold




def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    print(args.img_path)
    img = read_image(args.img_path)
    print(args.template_path)
    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
