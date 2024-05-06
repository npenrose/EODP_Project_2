import os
import sys


def main():
    try:
        from preprocessing import preprocessing
    except ImportError:
        print("Pre-processing function not found.")
        return
    preprocessing()


if __name__ == "__main__":
    main()