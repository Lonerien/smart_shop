import argparse


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Path to directory with files')
    parser.add_argument('-n', '--detector_names', type=str, required=True, help='Detector names')
    parser.add_argument('-w', '--detector_weights', type=str, required=True, help='Detector weights')
    parser.add_argument('-o', '--output', type=str, required=True, help='File to write final registry')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--prefill', type=str, default=None, help='File to prefill registry from')
    parser.add_argument('-v', '--verbose', type=bool, default=False, help='Enable verbose')

    args = parser.parse_args()
    return vars(args)
