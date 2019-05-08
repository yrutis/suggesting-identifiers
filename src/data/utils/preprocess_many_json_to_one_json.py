import pandas as pd

def main():
    df = pd.read_json('data/Android-Universal-Image-Loader.json', orient='records')

    df2 = pd.read_json('data/bigbluebutton.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/Bukkit.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/clojure.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/elasticsearch.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/junit.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/libgdx.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/metrics.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/netty.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/nokogiri.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/okhttp.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/platform_frameworks_base_1.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/platform_frameworks_base_2.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/presto.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/RxJava.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/spring-boot.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/spring-framework.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/storm.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    df2 = pd.read_json('data/zxing.json', orient='records')
    # print(df.head())
    print(len(df.index))
    df = df.append(df2, sort=True)

    # print(df.head())
    print(len(df.index))
    print(df.head())

    df2 = None

    df.to_json("all_methods_train.json", orient='records')


if __name__ == '__main__':
    main()
