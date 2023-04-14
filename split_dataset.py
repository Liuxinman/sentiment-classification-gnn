import pickle
import os

def split_dataset(fname):
    splited_data = {}
    splited_graph = {}
    splited_tree = {}

    fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    lines = fin.readlines()
    fin.close()
    fin = open(fname+'.graph', 'rb')
    idx2graph = pickle.load(fin)
    fin.close()
    fin = open(fname+'.tree', 'rb')
    idx2tree = pickle.load(fin)
    fin.close()

    total_samples = 1
    text_left, _, text_right = [s.lower().strip() for s in lines[0].partition("$T$")]
    aspect = lines[1].lower().strip()
    prev = text_left + " " + aspect + " " + text_right
    stack = [lines[0], lines[1], lines[2]]
    stack_graph = [idx2graph[0]]
    stack_tree = [idx2tree[0]]

    def add(num_aspects, stack, stack_graph, stack_tree):
        if num_aspects not in splited_data:
                splited_data[num_aspects] = []
                splited_graph[num_aspects] = []
                splited_tree[num_aspects] = []
        splited_data[num_aspects] += stack
        splited_graph[num_aspects]+= stack_graph
        splited_tree[num_aspects] += stack_tree

    for i in range(3, len(lines), 3):
        total_samples += 1
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text_raw = text_left + " " + aspect + " " + text_right
        if text_raw != prev:
            add(len(stack) // 3, stack, stack_graph, stack_tree)
            prev = text_raw
            stack, stack_graph, stack_tree = [], [], []

        stack += [lines[i], lines[i + 1], lines[i + 2]]
        stack_graph.append(idx2graph[i])
        stack_tree.append(idx2tree[i])
    if stack != []:
        add(len(stack) // 3, stack, stack_graph, stack_tree)

    new_num_samples = sum([len(splited_data[key]) // 3 for key in splited_data.keys()])
    new_graph_samples = sum([len(splited_graph[key]) for key in splited_graph.keys()])
    assert new_num_samples == total_samples, "dataset size does not match"
    assert new_num_samples == new_graph_samples, "dataset size and graph size does not match"

    sorted_num_samples = sorted(
        [(key, len(splited_data[key]) // (3 * key)) for key in splited_data.keys()],
        key=lambda x: x[0],
    )
    print(fname)
    for key, size in sorted_num_samples:
        print(f"number of aspects: {key}, number of samples: {size}")
    return splited_data, splited_graph, splited_tree


def save_to_txt(fpath, fname, splited_data, splited_graph, splited_tree):
    for key in splited_data.keys():
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        with open(f"{fpath}/{fname}.{key}", "w") as f:
            for line in splited_data[key]:
                f.write(line)
        idx2graph = {}
        idx2tree = {}
        j = 0
        for i in range(0, len(splited_data[key]), 3):
            idx2graph[i] = splited_graph[key][j]
            idx2tree[i] = splited_tree[key][j]
            j += 1
        pickle.dump(idx2graph, open(f"{fpath}/{fname}.{key}.graph", 'wb'))
        pickle.dump(idx2tree, open(f"{fpath}/{fname}.{key}.tree", 'wb'))


if __name__ == "__main__":
    dataset = "semeval14"
    fpath = f"./datasets/{dataset}"
    fnames = [
        "restaurant_train.raw",
        "restaurant_text.raw",
        "laptop_train.raw",
        "laptop_text.raw",
    ]
    for fname in fnames:
        splited_data, splited_graph, splited_tree = split_dataset(f"{fpath}/{fname}")
        save_to_txt(f"{fpath}/splited_{fname[:4]}", fname, splited_data, splited_graph, splited_tree)

