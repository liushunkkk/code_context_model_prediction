import os
import random
from os.path import join

import math
import networkx as nx
import xml.etree.ElementTree as ET
from gspan_mining.config import parser
from gspan_mining.main import main

import numpy as np
from networkx.algorithms import isomorphism

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code')


random.seed(23)

def get_models_by_ratio(project: str, start_ratio: float, end_ratio: float):
    """
    返回比例范围内的model，根据first_time排序

    :param project: 项目
    :param start_ratio: 开始比例
    :param end_ratio: 结束比例
    :return: model_dir数组
    """
    if start_ratio == end_ratio:
        return []
    project_path = join(root_path, project, 'repo_first_3')
    model_dir_list = os.listdir(project_path)
    all_models = []
    # random sample 352
    sample_dir_list = random.sample(model_dir_list, 352)
    print(sample_dir_list)
    for model_dir in sample_dir_list:
        model_path = join(project_path, model_dir)
        model_file = join(model_path, 'code_context_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        tree = ET.parse(model_file)  # 拿到xml树
        # 获取XML文档的根元素
        code_context_model = tree.getroot()
        first_time = code_context_model.get('first_time')
        all_models.append([model_dir, first_time])
    all_models = sorted(all_models, key=lambda x: x[1])
    m = np.array(all_models)
    return m[int(len(m) * start_ratio):int(len(m) * end_ratio), 0]


def load_patterns(patterns):
    G2s = []
    with open(patterns) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('t #'):
                g = nx.DiGraph()
            if line.startswith('v'):
                v = line.split(' ')
                g.add_node(int(v[1]), label=v[2])
            if line.startswith('e'):
                e = line.split(' ')
                g.add_edge(int(e[1]), int(e[2]), label=e[3])
            if line.startswith('#'):
                G2s.append(g)
        f.close()
    return G2s


def get_graph(graphs: list[ET.Element], step: int):
    gs = []
    if len(graphs) == 0:
        return gs
    seed_num_list = list()
    for graph in graphs:
        count = 0
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        for vertex in vertex_list:
            if vertex.get('seed') == '1':
                count += 1
        seed_num_list.append(count)
    one_seed_index = [i for i, x in enumerate(seed_num_list) if x == 1]
    remain_seed_list = list()
    for i in range(len(one_seed_index) - 1):
        remain_seed_list.append(math.floor((one_seed_index[i] + one_seed_index[i + 1]) / 2))
    remain_seed_list.append(math.ceil((one_seed_index[-1] + seed_num_list[-1]) / 2))
    for graph in graphs:
        vertices = graph.find('vertices')
        vertex_list = vertices.findall('vertex')
        edges = graph.find('edges')
        edge_list = edges.findall('edge')
        g = nx.DiGraph()
        true_node = 0
        # true_edge = 0
        # 转化为图结构
        for node in vertex_list:
            g.add_node(int(node.get('id')), label=node.get('stereotype'), origin=node.get('origin'),
                       seed=node.get('seed'))
            if int(node.get('origin')) == 1:
                true_node += 1
        for link in edge_list:
            g.add_edge(int(link.get('start')), int(link.get('end')), label=link.get('label'))
            # if int(link.get('origin')) == 1:
            #     true_edge += 1
        if graphs.index(graph) in remain_seed_list and true_node > 0:
            gs.append(g)
    return gs


def load_targets(project_model_name: str, step):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    G1s = []
    # 读取code context model
    model_dir_list = get_models_by_ratio(project_model_name, 0.84, 1)
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    for model_dir in model_dir_list:
        # print('---------------', model_dir)
        model_path = join(project_path, model_dir)
        model_file = join(model_path, f'{step}_step_seed_model.xml')
        # 如果不存在模型，跳过处理
        if not os.path.exists(model_file):
            continue
        # 读取code context model,以及doxygen的结果，分1-step,2-step,3-step扩展图
        tree = ET.parse(model_file)  # 拿到xml树
        code_context_model = tree.getroot()
        graphs = code_context_model.findall("graph")
        G1s = G1s + get_graph(graphs, step)
    return G1s


def count_positive(confidence, thres):
    count = 0
    for con in confidence:
        if con[1] >= thres:
            count += 1
    return count


def node_match(node1, node2):
    return node1['label'] == node2['label']


def edge_match(edge1, edge2):
    return edge1['label'] == edge2['label']


def calculate_result_full(labels, output, true_number):
    result = []
    for MinConf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if len(labels) == 0:
            result.append([0, 0, 0])
        positive_count = 0
        true_positive_count = 0
        for i in range(len(output)):
            if output[i] >= MinConf:
                positive_count += 1
                if labels[i] == 1:
                    true_positive_count += 1
        precision = true_positive_count / positive_count if positive_count != 0 else 0
        recall = true_positive_count / true_number
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        # print([precision, recall, f1])
        result.append([precision, recall, f1])
    return result


def print_result(result, k):
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for minConf in s:
        print(f'minConf: {minConf}:')
        i = s.index(minConf)
        p, r, f = 0.0, 0.0, 0.0
        for res in result:
            p += res[i][0]
            r += res[i][1]
            f += res[i][2]
        print(f'----------result of top {k}-------\n'
              f'Precision: {p / len(result)}, '
              f'Recall: {r / len(result)}, '
              f'F1: {f / len(result)}')


def graph_match(step, patterns):
    G1s = load_targets('my_mylyn', step)
    G2s = load_patterns(patterns)
    print('G1s', len(G1s), 'G2s', len(G2s))
    result_1, result_3, result_5, result_full = [], [], [], []
    for G1 in G1s:
        print(f'handling: {G1s.index(G1)}-{G1}')
        total_match = 0
        confidence = dict()
        for G2 in G2s:
            GM = isomorphism.DiGraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
            if GM.subgraph_is_isomorphic():
                for sub_iter in GM.subgraph_isomorphisms_iter():
                    total_match += 1
                    nodes = list(map(int, list(sub_iter.keys())))
                    for node in nodes:
                        if confidence.get(node):
                            confidence[node] = confidence[node] + 1
                        else:
                            confidence[node] = 1
                    # sub_G1: nx.DiGraph = G1.subgraph(nodes)
                    # print(sub_G1.nodes.data(), sub_G1.edges.data())
        for i in confidence:
            confidence[i] = confidence.get(i) / total_match
        confidence = sorted(confidence.items(), key=lambda d: d[1], reverse=True)
        # print(f'{G1} confidence {confidence}')
        # 去掉所有有关 seed 的节点
        new_confidence = []
        for top_c in confidence:
            node_id = top_c[0]
            if int(G1.nodes.get(node_id)['seed']) == 0:
                new_confidence.append(top_c)
        confidence = new_confidence
        output, labels = [], []
        for top_c in confidence:
            node_id = top_c[0]
            output.append(top_c[1])
            # print(G1.nodes.get(node_id))
            labels.append(int(G1.nodes.get(node_id)['origin']))
        true_number = 0
        for n in list(G1.nodes):
            true_number += int(G1.nodes.get(n)['origin'])
        result_full.append(calculate_result_full(labels, output, true_number))
    print_result(result_full, 0)


def graph_build_and_gspan(project_model_name='my_mylyn'):
    project_path = join(root_path, project_model_name, 'repo_first_3')
    graph_index = 0
    with open('./sample_graph.data', 'w') as f:
        # 读取code context model
        model_dir_list = get_models_by_ratio(project_model_name, 0, 0.84)
        print(len(model_dir_list))
        for model_dir in model_dir_list:
            # print('---------------', model_dir)
            model_path = join(project_path, model_dir)
            model_file = join(model_path, 'code_context_model.xml')
            # 如果不存在模型，跳过处理
            if not os.path.exists(model_file):
                continue
            # 读取code context model,以及doxygen的结果
            tree = ET.parse(model_file)  # 拿到xml树
            code_context_model = tree.getroot()
            graphs = code_context_model.findall("graph")
            for graph in graphs:
                graph_text = f't # {graph_index}\n'
                f.write(graph_text)
                vertices = graph.find('vertices')
                vertex_list = vertices.findall('vertex')
                vs = []
                for vertex in vertex_list:
                    stereotype, _id = vertex.get('stereotype'), vertex.get('id')
                    vs.append((_id, stereotype))
                for v in sorted(vs, key=lambda x: int(x[0])):
                    vertex_text = f'v {v[0]} {v[1]}\n'
                    f.write(vertex_text)
                edges = graph.find('edges')
                edge_list = edges.findall('edge')
                for edge in edge_list:
                    start, end, label = edge.get('start'), edge.get('end'), edge.get('label')
                    edge_text = f'e {start} {end} {label}\n'
                    f.write(edge_text)
                graph_index += 1
        f.write('t # -1')
        f.close()
    print(graph_index)

    min_support = math.ceil(0.007 * (graph_index - 1))  # 0.02 * num_of_graphs
    print('min_support: ', min_support)
    args_str = f'-s {min_support} ./sample_graph.data'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    main(FLAGS)


if __name__ == '__main__':
    graph_build_and_gspan()
    # graph_match(step=1, patterns='./new-patterns-0.007')