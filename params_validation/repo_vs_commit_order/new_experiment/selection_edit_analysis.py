"""
这里会根据计算出来的分位线来过滤异常值
保存到 IQR_code_timestamp 中
"""
import os
import shutil
import statistics
from os.path import join
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import mode

from xmlparser.doxygen_main import get_standard_elements

period_index = 0


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def read_xml_excel(url):
    """
    参数：
        url:文件路径
    """
    tree = ET.parse(url)  # 拿到xml树
    # 获取XML文档的根元素
    root = tree.getroot()
    events = root.find('code_elements')
    return events.get('total')


def get_all_code_timestamps():
    file_path = os.path.abspath(join(os.path.dirname(os.path.realpath(__file__)), '../', 'code_timestamp', '05'))
    print(file_path)
    project_list = os.listdir(file_path)
    # 进入项目目录
    project_to_xml_map = dict()
    for project_dir in project_list:
        if project_dir not in ['PDE', 'Mylyn', 'Platform', 'ECF']:
            continue
        bug_id_to_xml_map = dict()
        project_path = join(file_path, project_dir)
        xml_list = os.listdir(project_path)
        xml_list = [x[:x.find('.')] for x in xml_list]
        xml_list = sorted(xml_list, key=lambda x: int(x))
        xml_list = [x + ".xml" for x in xml_list]
        for xml_file in xml_list:
            xml_path = join(project_path, xml_file)
            tree = ET.parse(xml_path)  # 拿到xml树
            # 获取XML文档的根元素
            root = tree.getroot()
            bug_id = root.find('bug_id').text + "_" + root.find('id').text
            bug_id_to_xml_map[bug_id] = tree
        project_to_xml_map[project_dir] = bug_id_to_xml_map
    return project_to_xml_map


def add_event_kind_to_iqr_code_timestamp():
    """给 IQR 过滤之后的 working_periods 添加 event_kind 标签"""
    file_path = os.path.abspath(join(os.path.dirname(os.path.realpath(__file__)), '../', 'IQR_code_timestamp', '05'))
    print(file_path)
    project_list = os.listdir(file_path)
    project_to_xml_map = get_all_code_timestamps()
    # 进入项目目录
    for project_dir in project_list:
        if project_dir not in ['PDE', 'Mylyn', 'Platform', 'ECF']:
            continue
        project_path = join(file_path, project_dir)
        xml_list = os.listdir(project_path)
        xml_list = [x[:x.find('.')] for x in xml_list]
        xml_list = sorted(xml_list, key=lambda x: int(x))
        xml_list = [x + ".xml" for x in xml_list]
        bug_id_to_xml_map = project_to_xml_map[project_dir]
        for xml_file in xml_list:
            xml_path = join(project_path, xml_file)
            print(xml_path)
            tree = ET.parse(xml_path)  # 拿到xml树
            # 获取XML文档的根元素
            root_a = tree.getroot()
            bug_id = root_a.find('bug_id').text + "_" + root_a.find('id').text
            corresponding_xml_tree = bug_id_to_xml_map[bug_id]

            root_b = corresponding_xml_tree.getroot()

            # 查找所有 item 节点，并构建 b 的映射 {text: event_kind}
            def find_items_with_text(root):
                items = {}
                for elem in root.iter("element"):  # 查找所有 item 节点
                    text = elem.text.strip() if elem.text else ""  # 去除多余空白
                    if text:
                        items[text] = str(elem.get("event_kind"))
                return items

            b_mapping = find_items_with_text(root_b)

            # print(b_mapping)

            # 遍历 a，更新 event-type 属性
            for elem in root_a.iter("element"):  # 遍历 a 的所有 item 节点
                text = elem.text.strip() if elem.text else ""
                # print(text)
                if text in b_mapping:  # 如果 text 匹配
                    elem.set("event_kind", b_mapping[text])  # 更新 event_kind 属性
                else:
                    print("not find", text)

            # 输出更新后的 a
            tree.write(xml_path)


def add_event_kind_to_code_context_model():
    """给 code_context_model 添加 event_kind 标签"""
    file_path = os.path.abspath(join(os.path.dirname(os.path.realpath(__file__)), '../', 'IQR_code_timestamp', '05'))
    print(file_path)
    project_list = os.listdir(file_path)
    # 进入项目目录
    for project_dir in project_list:
        if project_dir not in ['PDE', 'Mylyn', 'Platform']:
            continue
        project_path = join(file_path, project_dir)
        xml_list = os.listdir(project_path)
        xml_list = [x[:x.find('.')] for x in xml_list]
        xml_list = sorted(xml_list, key=lambda x: int(x))
        xml_list = [x + ".xml" for x in xml_list]
        for xml_file in xml_list:
            xml_path = join(project_path, xml_file)
            print(xml_path)
            tree = ET.parse(xml_path)  # 拿到xml树
            # 获取XML文档的根元素
            root_a = tree.getroot()

            # code_context_model.xml
            model_path = os.path.abspath(
                join(os.path.dirname(os.path.realpath(__file__)), '../../', 'git_repo_code',
                     "my_" + project_dir.lower(), 'repo_first_3', xml_file[:xml_file.find(".")]))

            if not os.path.exists(model_path):
                continue

            model_tree = ET.parse(join(model_path, "_code_context_model.xml"))

            label_to_kind = dict()
            for elem in root_a.iter("element"):
                elem_text = elem.text.strip()
                qualified_name = get_standard_elements.solve_one(elem_text)[1]
                label_to_kind[qualified_name] = elem.get("event_kind")
            for vertex in model_tree.getroot().iter("vertex"):
                if vertex.get("label") not in label_to_kind:
                    print("not found", vertex.get("label"))
                else:
                    vertex.set("event_kind", label_to_kind[vertex.get("label")])

            model_tree.write(join(model_path, "_code_context_model.xml"))


def analysis_selection_edit_ratio(projects: list[str]):
    """分析 code_context_model 中 selection 和 edit 的标签分布"""
    file_path = os.path.abspath(join(os.path.dirname(os.path.realpath(__file__)), '../../', 'git_repo_code'))
    # 进入项目目录
    for project in projects:
        project_path = join(file_path, project, 'repo_first_3')
        model_dir_list = os.listdir(project_path)
        model_dir_list = sorted(model_dir_list, key=lambda x: int(x))
        s_e_r = []
        edit_0 = 0
        for model_dir in model_dir_list:
            model_path = join(project_path, model_dir)
            model_file = join(model_path, '_code_context_model.xml')
            xml_path = join(project_path, model_file)

            tree = ET.parse(xml_path)  # 拿到xml树
            # 获取XML文档的根元素
            root = tree.getroot()
            selection = 0.0
            edit = 0.0
            for elem in root.iter("vertex"):
                event_kind = elem.get("event_kind")
                if event_kind == "selection":
                    selection += 1
                elif event_kind == "edit":
                    edit += 1
            if edit == 0:
                edit_0 += 1
            else:
                s_e_r.append(selection / edit)

        print(f"----------项目: {project}------------")
        # 基本统计指标
        mean = np.mean(s_e_r)
        median = np.median(s_e_r)
        maximum = np.max(s_e_r)
        minimum = np.min(s_e_r)
        variance = np.var(s_e_r)
        std_dev = np.std(s_e_r)
        # 众数
        try:
            mode_val = statistics.mode(s_e_r)  # 单众数
        except statistics.StatisticsError:
            mode_val = "No unique mode"  # 如果有多个众数
        # 使用 scipy 计算多众数
        mode_result = mode(s_e_r, axis=None)
        # 分位数
        percentile_75 = np.percentile(s_e_r, 75)
        percentile_90 = np.percentile(s_e_r, 90)
        percentile_95 = np.percentile(s_e_r, 95)
        # 桶分布（直方图）
        bucket_counts, bin_edges = np.histogram(s_e_r, bins=10)  # 将数据分为4个桶

        # 打印结果
        print(f"model 总个数: {len(model_dir_list)}")
        print(f"edit 元素为 0 的 model 个数：{edit_0}")
        print(f"剩余 {len(model_dir_list) - edit_0} model 的比例统计指标(number_of_selection / number_of_edit):")
        print(f"平均值 (Mean): {mean:.2f}")
        print(f"中位数 (Median): {median:.2f}")
        print(f"最大值 (Max): {maximum:.2f}")
        print(f"最小值 (Min): {minimum:.2f}")
        print(f"方差 (Variance): {variance:.2f}")
        print(f"标准差 (Standard Deviation): {std_dev:.2f}")
        print("众数 (Mode):", mode_val)
        print("众数 (Scipy):", mode_result.mode, "出现次数:", mode_result.count)
        print("75分位数 (75th Percentile):", percentile_75)
        print("90分位数 (90th Percentile):", percentile_90)
        print("95分位数 (95th Percentile):", percentile_95)
        print("桶分布 (Bucket Distribution):")
        for i in range(len(bucket_counts)):
            print(f"范围 {bin_edges[i]:.2f} ~ {bin_edges[i + 1]:.2f}: {bucket_counts[i]}")


# add_event_kind_to_iqr_code_timestamp()
# add_event_kind_to_code_context_model()
analysis_selection_edit_ratio(['my_pde', 'my_platform', 'my_mylyn'])
