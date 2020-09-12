import os


if __name__ == '__main__':
    aspect_set = set()

    f = open(os.path.join('dataset', 'VLSP2018', 'VLSP2018-SA-Hotel-train.prod'), 'r')
    for line in f.read().split('\n\n'):
        lines = line.split('\n')
        asp = lines[1].split(',')[0].strip().lower()
        if asp not in aspect_set:
            aspect_set.add(asp)
    f.close()

    f = open(os.path.join('dataset', 'VLSP2018', 'VLSP2018-SA-Hotel-test.prod'), 'r')
    for line in f.read().split('\n\n'):
        lines = line.split('\n')
        asp = lines[1].split(',')[0].strip().lower()
        if asp not in aspect_set:
            aspect_set.add(asp)
    f.close()

    for item in aspect_set:
        print(item)
