import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the best epoch from log files.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing log files.")
    args = parser.parseewqw_args()


