# -*- coding: UTF-8 -*-

# 数据加载器基类
import struct


class Loader(object):
    '''
    数据加载器基类
    '''

    def __init__(self, path, count):
        '''
        初始化加载器
        :param path: 数据文件路径
        :param count: 文件中的样本个数
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        :return:
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        '''
        将unsigned byte字符转换为整数
        :param byte:
        :return:
        '''
        return struct.unpack('B', byte)[0]


class ImageLoader(Loader):
    '''
    # 图像数据加载器
    '''

    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        :param content:
        :param index:
        :return:
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(
                        content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        :param picture:
        :return:
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        :return:
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set


class LabelLoader(Loader):
    '''
    标签数据加载器
    '''

    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader(
        'G:\DRL\mnist\\train-images.idx3-ubyte',
        60000)
    label_loader = LabelLoader(
        'G:\DRL\mnist\\train-labels.idx1-ubyte',
        60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader(
        'G:\DRL\mnist\\t10k-images.idx3-ubyte',
        10000)
    label_loader = LabelLoader(
        'G:\DRL\mnist\\t10k-labels.idx1-ubyte',
        10000)
    return image_loader.load(), label_loader.load()
