import sys
import argparse
import coelurus
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='Specify your config.ini with configuration.')


def main():
    args = parser.parse_args()
    config_path = args.config_path
    loader = coelurus.Loader(config_path)
    loader.load_data()

    val = coelurus.Validator(loader)
    val.enforce_column_names()
    val.quality_check()

    data_filter = coelurus.DataPrepare(val)
    data_filter.apply_filters()


if __name__ == "__main__":
    main()
