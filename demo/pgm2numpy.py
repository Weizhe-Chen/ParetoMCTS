import re
import numpy as np


def read_pgm(filename, byteorder=">"):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    This code snippet is taken from:
    https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm

    """
    with open(filename, "rb") as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)",
            buffer,
        ).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(
        buffer,
        dtype="u1" if int(maxval) < 256 else byteorder + "u2",
        count=int(width) * int(height),
        offset=len(header),
    ).reshape((int(height), int(width)))
