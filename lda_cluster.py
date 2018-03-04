#-*- coding:utf-8 -*-
import logging
import logging.config
import os
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from nltk.parse import stanford
from nltk.parse.stanford import StanfordDependencyParser
import scipy.cluster.hierarchy as sch
import title_parser
import numpy

#获取当前路径
path = os.getcwd()
#导入日志配置文件
logging.config.fileConfig("logging.conf")
#创建日志对象
logger = logging.getLogger()

#层次聚类
def hierarchy(vectors, cluster_numb):
    disMat = sch.distance.pdist(vectors, 'euclidean')
    Y = sch.linkage(disMat, method="ward")
    _, ax = plt.subplots(figsize=(20, 80))
    Z = sch.dendrogram(Y, orientation='right')
    ax.set_xticks([])
    ax.set_title('应用研究聚类', fontproperties=FontProperties(fname="huawenfansong.ttf"),
                       fontsize=20)
    ax.set_yticklabels(np.array(range(len(vectors))), fontproperties=FontProperties(fname="huawenfansong.ttf"),
                       fontsize=12)
    ax.set_frame_on(False)
    plt.savefig("cluster.png")
    # 根据linkage matrix Y得到聚类结果:
    cluster = sch.fcluster(Y,t=cluster_numb,criterion='maxclust')
    print "Original cluster by hierarchy clustering\n"
    return cluster

#输入一行以空格隔开的文本统计词频
def wordcount(line):
    count = {}
    for i in line.split(' '):
        if i not in count.keys():
            count[i] = 1
        else:
            count[i] += 1
    top_words = sorted(count.keys(),key= lambda x:count[x],reverse=True)
    sorted_result = []
    for i in top_words:
        if len(i) < 3:
            continue
        sorted_result.append([i,count[i]])
    return sorted_result


if __name__ == '__main__':
    theta_vector = []
    paper = []

    #读入文献
    with open('data\\yingyongyanjiu_titles.dat','r') as paper_f:
        for line in paper_f.readlines():
            paper.append(line.strip())
    logger.info(u"读入文献数据。")
    #读lda数据
    with open('data\\tmp\\model_theta.dat','r') as theta_f:
        for line in theta_f.readlines():
            vector = [float(x) for x in line.strip().split('\t')]
            theta_vector.append(vector)
    logger.info(u"lda预训练数据载入完成。")


    #层次聚类
    logger.info(u"层次聚类进行...")
    cluster_numb = 9
    cluster_result = hierarchy(theta_vector,cluster_numb)

    cluster_dict = {}
    for indx,i in enumerate(cluster_result):
        if i not in cluster_dict.keys():
            cluster_dict[i] = [indx]
        else:
            cluster_dict[i].append(indx)
    sorted_cluster = sorted(cluster_dict.keys(),key=lambda x:cluster_dict[x],reverse=True)

    logger.info(u"层次聚类完成。")
    logger.info(u"载入standford句法分析器")
    '''领域词和方法词切分'''
    os.environ['STANFORD_PARSER'] = 'E:/NLP/stanfordNLTK/stanfordParser/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = 'E:/NLP/stanfordNLTK/stanfordParser/stanford-parser-3.6.0-models.jar'
    eng_parser = StanfordDependencyParser(model_path="E:/NLP/stanfordNLTK/chinesePCFG.ser.gz", encoding="gb2312")
    words_tags = title_parser.read_dict('relation_dict/words_tag.txt')
    relation_tags = title_parser.read_dict('relation_dict/relation_tag.txt')
    useless_wordtags = ['DT', 'CC']
    useless_relationtags = ['nummod']

    result = {}
    logger.info(u"对层次聚类%s个簇进行处理..." % cluster_numb)




    for cluster_indx in sorted_cluster:
        cluster_papers = [paper[id] for id in cluster_dict[cluster_indx]]

        theta_npVector = numpy.array([theta_vector[x] for x in cluster_dict[cluster_indx]], dtype=numpy.float64)
        centroid_v = numpy.zeros(len(theta_vector[0]))
        for indx in cluster_dict[cluster_indx]:
            centroid_v = centroid_v + numpy.array(theta_vector[indx])
        centroid_v = centroid_v/len(theta_npVector)

        cluster_dist = 0
        for vec in theta_npVector:
            dist = numpy.linalg.norm(vec - centroid_v)
            cluster_dist = cluster_dist + dist
        logger.info(u"第%s个类，共%s篇文献,类内距为：%s" % (cluster_indx,len(cluster_papers),str(cluster_dist/len(cluster_papers))))



        # research_filed = []
        # method = []
        # line = ""
        # logger.info(u"第%s个类词级层次关系处理...." % cluster_indx)
        # for indx,p in enumerate(cluster_papers):
        #     print indx+1,"/", len(cluster_papers)
        #     l = title_parser.dependency_parser(p,eng_parser,words_tags,relation_tags,useless_wordtags,useless_relationtags)
        #     if len(l) == 6:
        #         if len(l[4]) > 0:
        #             research_filed.extend(l[4].split(','))
        #         if len(l[5]) > 0:
        #             method.extend(l[5].split(','))
        #         line += l[1] + " "
        # count = wordcount(line)

        # result[str(cluster_indx)] = {"cluster_size": len(cluster_papers),
        #                              "word_frequency": [str(x[0]) + ',' + str(x[1]) for x in count],
        #                              "cluster_paper": cluster_papers,
        #                              "cluster_name": "",
        #                              "filed_words": list(set(research_filed)),
        #                              "method_words": list(set(method))
        #                              }

    # '''记录结果到文件中'''
    # logger.info(u"保存数据")
    # with open("./yingyongyanjiu.json", 'w') as json_file:
    #     json.dump(result, json_file, ensure_ascii=False)
    # logger.info(u"完成")



