import os


def main(path):
    for pic in os.listdir(path):
        print(pic[:-4])

if __name__ == '__main__':
    path = 'photos'
    main(path)
