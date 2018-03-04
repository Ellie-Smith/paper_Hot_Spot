#coding=utf-8
import os
import sys
import csv
import re
reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import time
from nltk.parse import stanford
from nltk.parse.stanford import StanfordDependencyParser


class Fileds:
    def __init__(self):
        self.sub_filed = None
        self.words = []
    def add_word(self,word):
        self.words.append(word)
    def delete_word(self,word):
        if word in self.words:
            self.words.remove(word)
            return True
        else:
            return False

def read_csv(filename):
    import csv
    csvFile = open(filename, "r")
    reader = csv.reader(csvFile)
    return reader

def read_dict(filename):
    result_dict = {}
    with open(filename) as f:
        for i in f.readlines():
            pare = i.strip('\n').strip('\r\n').split(":")
            if len(pare) >= 2:
                result_dict[pare[0]] = "@"+pare[1]+"@"
            else:
                continue
    return result_dict

#根据给出的父节点返回子节点
def get_current_filed(parent_node,parent_words,child_words,candidates,union):
    node = Fileds()
    c = 0
    for w in parent_node.words:
        for indx,tar in enumerate(parent_words):
            if w == tar and child_words[indx] in candidates:
                node.add_word(child_words[indx])
                c += 1
                #把相应的联合词放在同一级
                u = [x[1] for x in union if x[0] == child_words[indx]]
                for i in u:
                    node.add_word(i)
                    c += 1

    return node,c

#对有如下关系的归为同一类
def find_specific_relation(cur_filed_words,relation_m):
    finding_relation = ["dobj","amod","prep","nsubj","nn"]
    for row in relation_m:
        if row[0][0] in cur_filed_words and row[1] in finding_relation:
            return True
    return False



#分好领域词和方法词后的最后合并（名词组合形式）
def find_nn(words,relation):
    result = list(set(words))
    for w in set(words):
        for row in relation:
            if row[1] == 'nn' and w == row[0][0] and row[2][0] in words:
                result.append(row[2][0] + row[0][0])
                if row[0][0] in result: result.remove(row[0][0])
                if row[2][0] in result: result.remove(row[2][0])
    return result

'''领域词方法词划分，返回
        result_list
        [ title ,
          split_title,
          relation,
          filed_map,
          research_filed,
          method
        ]
'''
def dependency_parser(title, eng_parser, words_tags, relation_tags, useless_wordtags, useless_relationtags):
    sub_result = []
    print title
    sub_result.append(title)

    '''
                    去除无用的高频词
    '''
    useless_word = [w[0] for w in read_csv("data\\useless_words.csv")]


    split_titles = ' '.join(jieba.cut(title))

    #计时器
    begin = long(time.time())
    try:
        res = list(eng_parser.parse(split_titles.split()))
        relation_map = res[0].triples()  # 用于表示词汇间关系
        # 二次分析，发现新词
        for i in relation_map:
            if i[1] == 'nn':
                jieba.add_word(i[2][0] + i[0][0])
                jieba.add_word(i[2][0] + ' ' + i[0][0])
        res = list(eng_parser.parse(' '.join(jieba.cut(title)).split()))
        sub_result.append(' '.join(jieba.cut(title)))
        print ' '.join(jieba.cut(title))

        relation_map = res[0].triples()
        in_out_count = {}  # 用于计算每个词的出度和入度

        parents = []
        children = []
        union = []
        # 计算词语的度
        relation = ""
        new_relation_map = []
        for row in relation_map:
            if row[1] in useless_relationtags: continue
            # 依赖关系按词语出现顺序调整顺序
            if row[1] == 'dep' or row[1] == 'dobj':

                numb1 = title.find(row[0][0], 0)
                numb2 = title.find(row[2][0], 0)
                if numb1 < numb2:
                    row = (row[2], row[1], row[0])
            new_relation_map.append(row)
            # 出度都需要计算，出度表示父节点，入度表示子节点
            if row[0][0] not in in_out_count.keys():
                in_out_count[row[0][0]] = {'in': 0, 'out': 1}
            else:
                in_out_count[row[0][0]]['out'] += 1

            tag1 = row[0][1]
            tag2 = row[1]
            tag3 = row[2][1]
            if tag1 in words_tags.keys():
                tag1 = words_tags[tag1]
            if tag3 in words_tags.keys() and tag3 not in useless_wordtags:
                tag3 = words_tags[tag3]
                if row[1] != 'conj':
                    parents.append(row[0][0])
                    children.append(row[2][0])
                else:
                    union.append([row[0][0], row[2][0]])
                for u in union:
                    if u[0] in children and u[1] not in children:
                        children.append(u[1])
                    elif u[1] in children and u[0] not in children:
                        children.append(u[0])
                # 计算入度，只有转译成功的词语才计算入度
                if row[2][0] not in in_out_count.keys():
                    in_out_count[row[2][0]] = {'in': 1, 'out': 0}
                else:
                    in_out_count[row[2][0]]['in'] += 1
            if tag2 in relation_tags.keys():
                tag2 = relation_tags[tag2]
            relation += str(row[0][0]) + ' ' + str(tag1) + ' , ' + str(tag2) + ' , ' + str(row[2][0]) + ' ' + str(
                    tag3) + ';'
            print row[0][0], tag1, ',', tag2, ',', row[2][0], tag3
            if tag2 == 'nn':  # 新词发现
                relation += str(row[2][0]) + str(row[0][0]) + ';'
                print row[2][0] + row[0][0]
                jieba.add_word(row[2][0] + row[0][0])
        sub_result.append(relation)

        # 剔除转译失败的pair,构建词汇父子关系和联合关系
        # relation_map = [x for x in relation_map if x[2][1] in words_tags.keys()]
        # parents = [w[0][0] for w in relation_map if w[1] != 'conj']
        # children = [w[2][0] for w in relation_map if w[1] != 'conj']
        # union = [[w[0][0],w[2][0]] for w in relation_map if w[1] == 'conj']
        # print 'parent:',','.join(parents)
        # print 'children:',','.join(children)
        # print 'union:',','.join([s[0]+'-'+s[1] for s in union])

        # 按词汇的度排序（出度权重为2，入度权重为1），并去除总度数小于2的词汇
        candidate = sorted(in_out_count.keys(), key=lambda x: in_out_count[x]['in'] + 2 * in_out_count[x]['out'],
                                     reverse=True)
        candidate_words = [w for w in candidate if in_out_count[w]['in'] + 2 * in_out_count[w]['out'] >= 1]
        # print 'candidate_word:',','.join(candidate_words)
        # 构建领域词汇等级链表

        root = Fileds()
        count = 0
        for w in candidate_words:
            if w not in children:
                root.add_word(w)
                count += 1
        cur = root
        # print "root:",','.join(root.words)
        while count < len(candidate_words) or cur != None:
            sub_node, c = get_current_filed(cur, parent_words=parents, child_words=children, candidates=candidate_words,
                                                union=union)
            end = long(time.time())
            if end-begin > 20:
                raise RuntimeError('TimeOutError')
            if len(sub_node.words) > 0:
                cur.sub_filed = sub_node
                count += c

            cur = cur.sub_filed
        cur = root

        '''将层级词汇打印出来'''
        layers = 0
        fileds = ""
        while cur != None:
            cur_filed_word = [w for w in set(cur.words) if w not in useless_word]
            cur_filed = '[' + ','.join(cur_filed_word) + ']' + "<--"

            end = long(time.time())
            if end - begin > 20:
                raise RuntimeError('TimeOutError')

            if len(cur_filed_word) > 0:
                layers += 1
                print cur_filed,
                fileds += cur_filed
            cur = cur.sub_filed
        print "层数： ", layers
        sub_result.append(fileds)

        '''将分层的词语划分为研究领域词和方法词'''
        cur_filed = root
        while cur_filed != None and len([w for w in cur_filed.words if w not in useless_word]) == 0:
            cur_filed = cur_filed.sub_filed
            end = long(time.time())
            if end - begin > 20:
                raise RuntimeError('TimeOutError')
        research_filed = [w for w in cur_filed.words if w not in useless_word]
        method = []

        # 第一层必为领域层
        if layers == 2:  # 两层的分别为领域层和方法层
            method.extend([w for w in cur_filed.sub_filed.words if w not in useless_word])
        elif layers > 2:  # 三层以上的,看领域词是否有直接宾语、形容词修饰、介词修饰、名词性主语关系且右边是下一层词语
            while find_specific_relation(cur_filed.words, new_relation_map):
                end = long(time.time())
                if end - begin > 20:
                    raise RuntimeError('TimeOutError')
                research_filed.extend([w for w in cur_filed.sub_filed.words if w not in useless_word])
                cur_filed = cur_filed.sub_filed
            # 之后的都归为方法层
            cur_filed = cur_filed.sub_filed
            while cur_filed != None and len(cur_filed.words) != 0:
                end = long(time.time())
                if end - begin > 20:
                    raise RuntimeError('TimeOutError')
                method.extend([w for w in cur_filed.words if w not in useless_word])
                cur_filed = cur_filed.sub_filed

        '''再把两部分词语中名词组合关系的合并'''
        research_filed = find_nn(research_filed, new_relation_map)
        method = find_nn(method, new_relation_map)

        print "领域词：", ','.join(research_filed)
        print "方法词：", ','.join(method)
        sub_result.append(','.join(research_filed))
        sub_result.append(','.join(method))
        print ''
        print ''
        return sub_result

    except Exception, e:
        print e
        print ''
        return []
        pass










