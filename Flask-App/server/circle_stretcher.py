import math
import collections.abc
import numpy

# https://squircular.blogspot.com/2015/09/mapping-circle-to-square.html
def _elliptical_to_square(x, y):
    try:
        return x * math.sqrt(1.0 - y * y / 2.0), y * math.sqrt(1.0 - x * x / 2.0)
    except ValueError:  # sqrt of a negative number
        return None
    
def _get_zero_pixel_value(pixel):
    if isinstance(pixel, collections.abc.Iterable):
        return type(pixel)(0 for _ in pixel)
    return 0

def _pixel_coordinates_to_unit(coordinate, max_value):
    return coordinate / max_value * 2 - 1

def _one_coordinates_to_pixels(coordinate, max_value):
    return (coordinate + 1) / 2 * max_value

def _zeros_like(inp):
    if isinstance(inp, numpy.ndarray):
        return numpy.zeros_like(inp)

    zero = _get_zero_pixel_value(inp[0][0])
    return [[zero] * len(inp) for _ in inp]

def _check_that_all_sides_are_the_same_length(inp):
    for x, row in enumerate(inp):
        if len(row) != len(inp):
            raise ValueError(
                f"The input image must be square shaped but row {x} "
                f"is {len(row)} pixels accross, while the other side of the "
                f"image is {len(inp)}"
            )

def to_square(disk):
    _check_that_all_sides_are_the_same_length(disk)

    result = _zeros_like(disk)  # an black image wit the same dimensions 

    for x, row in enumerate(disk):
        # x and y are in the range(0, len(inp)) but they need to be between -1 and 1
        # for the code
        unit_x = _pixel_coordinates_to_unit(x, len(disk))

        for y, _ in enumerate(row):
            unit_y = _pixel_coordinates_to_unit(y, len(row))

            try:
                uv = _elliptical_to_square(unit_x, unit_y)
                if uv is None:
                    continue
                u, v = uv

                u = _one_coordinates_to_pixels(u, len(disk))
                v = _one_coordinates_to_pixels(v, len(row))

                # TODO: something smarter than flooring.
                # maybe take a weighted average of the nearest 4 pixels
                result[x][y] = disk[math.floor(u)][math.floor(v)]
            except IndexError:
                pass

    return result
