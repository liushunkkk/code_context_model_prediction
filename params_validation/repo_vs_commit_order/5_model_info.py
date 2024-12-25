"""
用于统计context model的节点和边以及连通分量的数据
"""
import os
import shutil

import networkx as nx
from os.path import join
import xml.etree.ElementTree as ET

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'git_repo_code')


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_graph(vertices: list[ET.Element], edges: list[ET.Element]):
    g = nx.DiGraph()
    # 转化为图结构
    for node in vertices:
        g.add_node(node.get('id'))
    for link in edges:
        g.add_edge(link.get('start'), link.get('end'))
    return g


def main_func(project_name: str, mode: str):
    project_path = join(root_path, project_name)
    info_dir_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), f'{mode}model_info')
    # make_dir(info_dir_path)
    strategy_dir_list = os.listdir(project_path)
    for strategy_dir in strategy_dir_list:
        strategy_path = join(project_path, strategy_dir)
        info_strategy_path = join(info_dir_path, strategy_dir)
        make_dir(info_strategy_path)
        model_dir_list = os.listdir(strategy_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        info_root = ET.Element("context_model_info")
        model_total = 0
        component_root = 0
        for model_dir in model_dir_list:
            model_path = join(strategy_path, model_dir, '_code_context_model.xml')
            if not os.path.exists(model_path):
                continue
            model_total += 1
            info_model = ET.SubElement(info_root, 'code_context_model')
            info_model.set('id', model_dir)
            tree = ET.parse(model_path)  # 拿到xml树
            # 获取XML文档的根元素
            root = tree.getroot()
            graphs = root.findall("graph")
            node_total = 0
            edge_total = 0
            component_total = 0
            for graph in graphs:
                info_graph = ET.SubElement(info_model, 'graph')
                vertices = graph.find('vertices').findall('vertex')
                edges = graph.find('edges').findall('edge')
                net_graph = get_graph(vertices, edges)
                connected_components = list(nx.weakly_connected_components(net_graph))
                node_count = 0
                edge_count = 0
                for cc in connected_components:
                    subgraph = net_graph.subgraph(cc)
                    # print('subgraph', subgraph, subgraph.nodes(), subgraph.edges())
                    info_component = ET.SubElement(info_graph, 'connected_component')
                    info_component.set('node_num', str(len(subgraph.nodes())))
                    info_component.set('edge_num', str(len(subgraph.edges())))
                    info_component.set('nodes', str(subgraph.nodes()))
                    info_component.set('edges', str(subgraph.edges()))
                    un_subgraph = subgraph.to_undirected()
                    info_component.set('diameter', str(nx.diameter(un_subgraph)))
                    node_count += len(subgraph.nodes())
                    edge_count += len(subgraph.edges())
                info_graph.set('nodes', str(node_count))
                info_graph.set('edges', str(edge_count))
                info_graph.set('components', str(len(connected_components)))
                node_total += node_count
                edge_total += edge_count
                component_total += len(connected_components)
            info_model.set('nodes', str(node_total))
            info_model.set('edges', str(edge_total))
            info_model.set('components', str(component_total))
            component_root += component_total
        info_root.set('models', str(model_total))
        info_root.set('components', str(component_root))
        info_tree = ET.ElementTree(info_root)
        # 将XML写入文件
        xml_file_name = os.path.join(info_strategy_path, project_name + ".xml")
        info_tree.write(xml_file_name)


def remove_empty_dir(project_model_name: str):
    """
    删除没有model目录，跑每个项目的extract之前跑一下就行了，

    :param project_model_name: 项目目录名字
    """
    project_path = join(root_path, project_model_name)
    strategy_dir_list = os.listdir(project_path)
    for strategy_dir in strategy_dir_list:
        strategy_path = join(project_path, strategy_dir)
        model_dir_list = os.listdir(strategy_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        for model_dir in model_dir_list:
            print('---------------', model_dir)
            model_path = join(strategy_path, model_dir)
            model_file = join(model_path, 'code_context_model.xml')
            # 如果不存在模型，跳过处理
            if not os.path.exists(model_file):
                print('not exist', model_file)
                shutil.rmtree(model_path)


# if want see info before process set mode to '', else set 'final_'
mode = 'final_'
# remove_empty_dir('my_ecf')
# main_func('my_ecf', mode=mode)
# remove_empty_dir('my_pde')
# main_func('my_pde', mode=mode)
# # remove_empty_dir('my_platform')
# main_func('my_platform', mode=mode)
# # remove_empty_dir('my_mylyn')
# main_func('my_mylyn', mode=mode)



mode = 'final_'
project_name = 'my_mylyn'
first_times = []
last_times = []

project_path = join(root_path, project_name)
info_dir_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), f'{mode}model_info')
# make_dir(info_dir_path)
strategy_dir_list = os.listdir(project_path)
for strategy_dir in strategy_dir_list:
    strategy_path = join(project_path, strategy_dir)
    info_strategy_path = join(info_dir_path, strategy_dir)
    make_dir(info_strategy_path)
    model_dir_list = os.listdir(strategy_path)
    model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
    info_root = ET.Element("context_model_info")
    model_total = 0
    component_root = 0
    for model_dir in model_dir_list:
        model_path = join(strategy_path, model_dir, 'code_context_model.xml')
        if not os.path.exists(model_path):
            continue
        model_total += 1
        info_model = ET.SubElement(info_root, 'code_context_model')
        info_model.set('id', model_dir)
        tree = ET.parse(model_path)  # 拿到xml树
        # 获取XML文档的根元素
        root = tree.getroot()
        first_times.append(root.get('first_time'))
        last_times.append(root.get('last_time'))
        # print(root.get('first_time') == root.get('last_time'))
print(min(first_times), max(first_times))
print(min(last_times), max(last_times))