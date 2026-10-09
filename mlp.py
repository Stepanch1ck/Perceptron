import sys

import split
import train
import predict


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ('split', 'train', 'predict'):
        subcmd = sys.argv[1].lower()
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        if subcmd == 'split':
            split.main()
        elif subcmd == 'train':
            train.main()
        elif subcmd == 'predict':
            predict.main()
    else:
        train.main()


if __name__ == '__main__':
    main()
