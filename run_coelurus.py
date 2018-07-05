import sys
import argparse
from coelurus import Loader
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='Specify your config.ini with configuration.')

def main():
    args = parser.parse_args()
    config_path = args.config_path
    loader = Loader(config_path)
    loader.load_data()
    print(loader.input_data)


if __name__ == "__main__":
    main()
