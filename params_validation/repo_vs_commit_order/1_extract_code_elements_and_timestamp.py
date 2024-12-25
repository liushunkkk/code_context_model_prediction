"""
这个文件用于提取各个working periods的code elements和他们的timestamp
"""
import os
import shutil
from os.path import join
import numpy as np
import xml.etree.ElementTree as ET


def make_dir(directory):
    """
    创建一个目录

    :param directory: 目录地址
    :return: 无返回值，创建目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def read_xml(url):
    """
    读取xml格式文件

    参数：
        url:文件路径
    返回：
        data:表格中的数据
    """
    tree = ET.parse(url)  # 拿到xml树
    # 获取XML文档的根元素
    root = tree.getroot()
    events = root.findall('event')
    data = []
    # 遍历每一行数据
    for event in events:
        # 定义表格存储每一行数据
        da = [event.get("event_kind"), event.get("event_structure_kind"),
              event.get("event_structure_handle"), event.get("event_start_date"), event.get("event_kind")]
        # 存储每一行数据
        data.append(da)
    timestamp = root.find('timestamp')
    # 返回数据
    return str(root.get('bug_id')), data, [timestamp.get('start'), timestamp.get('last')]


def main_fun():
    origin_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'working_periods', 'code_elements')
    origin_file_dir_list = os.listdir(origin_path)
    for index_dir in origin_file_dir_list:
        if index_dir not in ['05']:
            continue
        index_path = join(origin_path, index_dir)
        project_list = os.listdir(index_path)
        # 进入项目目录
        for project_dir in project_list:
            if project_dir not in ['PDE', 'Mylyn', 'Platform', 'ECF']:
                continue
            make_dir("code_timestamp/" + index_dir + '/' + project_dir)
            project_path = join(index_path, project_dir)
            xml_list = os.listdir(project_path)
            xml_list = [x[:x.find('.')] for x in xml_list]
            xml_list = sorted(xml_list, key=lambda x: int(x))
            xml_list = [x + ".xml" for x in xml_list]
            for period_xml_file in xml_list:
                bug_id, xml_data, time_stamp = read_xml(join(project_path, period_xml_file))  # 拿到所有的data
                a = np.array(xml_data)
                # 得到不重复的code elements
                con = []
                handles = []  # 用来保存handles
                for line in xml_data:
                    if line[2] in handles:
                        con.append(False)
                    else:
                        handles.append(line[2])
                        con.append(True)
                effective_data = a[con]  # 得到有效的event
                if len(effective_data) <= 0:
                    continue
                else:
                    # 创建根元素
                    root = ET.Element("working_period")
                    # 设置id
                    period_id = ET.SubElement(root, "id")
                    period_id.text = str(period_xml_file[:period_xml_file.find('.xml')])
                    # 设置项目
                    project_et = ET.SubElement(root, "project")
                    project_et.text = str(project_dir)
                    # 设置bug_id
                    project_et = ET.SubElement(root, "bug_id")
                    project_et.text = bug_id
                    # 设置时间戳
                    timestamps = ET.SubElement(root, "timestamps")
                    first_stamp = ET.SubElement(timestamps, "first")
                    first_stamp.text = time_stamp[0]
                    last_stamp = ET.SubElement(timestamps, "last")
                    last_stamp.text = time_stamp[1]
                    # 设置code elements
                    code_elements = ET.SubElement(root, "code_elements")
                    code_elements.set("total", str(len(effective_data)))
                    for line in effective_data:
                        element = ET.SubElement(code_elements, "element")
                        element.set('event_start_date', line[3])
                        element.set('event_kind', line[4])
                        element.text = line[2]
                    # 创建XML树
                    tree = ET.ElementTree(root)
                    # 将XML写入文件
                    xml_file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 "code_timestamp/", index_dir, project_dir, period_xml_file)
                    tree.write(xml_file_name)
                    print("{0} working period of {1}'s {2} bug has {3} valid code elements".format(period_xml_file,
                                                                                                   project_dir,
                                                                                                   bug_id,
                                                                                                   len(effective_data)))


main_fun()
