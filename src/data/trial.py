import pickle

def main():
    with open('{}.dict.c2s'.format('java-small'), 'rb') as file:
        subtoken_to_count = pickle.load(file)
        print(subtoken_to_count)
        print(len(subtoken_to_count))


if __name__ == '__main__':
    main()