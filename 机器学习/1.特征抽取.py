# -*- coding:utf-8 -*-

# author: pcw
# datetime: 2018/12/4 2:14 PM
# software: PyCharm

# 使用csv文件保存数据，pandas，numpy，TensorFlow

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import jieba


def dictves():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    # 字典数据抽取
    data = dict.fit_transform([{'city': '北京', 'tem': 100}, {'city': '上海', 'tem': 60}, {'city': '深圳', 'tem': 30}])

    print(data)
    print(dict.inverse_transform(data))
    print(dict.get_feature_names())
    """
    one-hot 编码
    [[  0.   1.   0. 100.]
     [  1.   0.   0.  60.]
     [  0.   0.   1.  30.]]
    [{'city=北京': 1.0, 'tem': 100.0}, {'city=上海': 1.0, 'tem': 60.0}, {'city=深圳': 1.0, 'tem': 30.0}]
    ['city=上海', 'city=北京', 'city=深圳', 'tem']
    """
    return None


def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["life is short,i like python", "life life is too long,i dislike python"])
    # 中文需要先分词
    data = cv.fit_transform(["人生苦短，我喜欢python", "人生漫长，不用python"])

    # 统计所有文章红所有的词，重复的只看做一次。词的列表，单个字母不统计
    # 对每篇文章统计所有词的出现的次数

    print(data.toarray())
    print(cv.get_feature_names())

    return None


def cutword():
    con1 = jieba.cut('今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。')
    con2 = jieba.cut('我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。')
    con3 = jieba.cut('如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。')

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    :return: None
    TF-IDF 重要性程度
    tf:词的频率
    idf:逆文档频率  log(总文档数量/该词出现的文档数量) 值越小，重要性越高
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)

    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())
    print(data.toarray())

    return None

def tfidfvec():
    """
    中文特征值化
    :return: None
    TF-IDF 重要性程度
    tf:词的频率
    idf:逆文档频率  log(总文档数量/该词出现的文档数量) 值越小，重要性越高
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())
    print(data.toarray())

    return None




if __name__ == '__main__':
    # dictves()
    # countvec()
    # hanzivec()
    tfidfvec()
