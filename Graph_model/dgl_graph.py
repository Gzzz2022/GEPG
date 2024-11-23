import os
import dgl


def dgl_graph(data_name: str, type: str, node: str):
    g = dgl.DGLGraph()

    g.add_nodes(node)
    edge_list = []
    if type == 'direct':
        with open(f'./data/{data_name}/graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)

        return g
    elif type == 'undirect':
        with open(f'./data/{data_name}/graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)

        g.add_edges(dst, src)
        return g
    elif type == 'e_to_k':
        with open(f'./data/{data_name}/graph/e_to_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'k_to_e':
        with open(f'./data/{data_name}/graph/k_to_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_to_u':
        with open(f'./data/{data_name}/graph/e_to_u.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_to_e':
        with open(f'./data/{data_name}/graph/u_to_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
