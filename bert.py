import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import argparse #设置参数
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
# Bert
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm
import argparse

#%matplotlib inline
#%run ./Metrics.py

if torch.cuda.is_available():
    device = 'cuda:0' 
else:
    device = 'cpu'

def remove_punc_stopword(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    remove_punc = [word for word in text if word not in string.punctuation]
    remove_punc = ''.join(remove_punc)
    return [word.lower() for word in remove_punc.split() if word.lower() not in stopwords.words('english')]

def categorize_Sentiment(sentiment):
    if sentiment =='Positive' :
        return 2
    elif sentiment == 'Neutral':
        return 1
    elif sentiment == 'Negative':
        return 0
    else:
        return np.nan  


class BERTClassifier(nn.Module):
    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self, output_dim, pretrained_name='bert-base-uncased'):
        super(BERTClassifier, self).__init__()
        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained(pretrained_name)
        # 外接全连接层
        self.mlp = nn.Linear(768, output_dim)
    def forward(self, tokens_X):
        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        res = self.bert(**tokens_X)
        # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析 
        return self.mlp(res[1])

"""
评估函数，用以评估数据集在神经网络下的精确度
"""
def evaluate(net, batch, tokenizer,comments_data, labels_data):
    sum_correct, i = 0, 0
    batch=batch
    while i <= len(comments_data):
        comments = comments_data[i: min(i + batch, len(comments_data))]
        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device=device)
        res = net(tokens_X)                                          # 获得到预测结果
        y = torch.tensor(labels_data[i: min(i + batch, len(comments_data))]).reshape(-1).to(device=device)
        sum_correct += (res.argmax(axis=1) == y).sum()              # 累加预测正确的结果
        i += batch
    return sum_correct/len(comments_data)                           # 返回(总正确结果/所有样本)，精确率

def main(args):
    #review_data=pd.read_csv(r"C:\Users\Chuxu\ESE527\googleplaystore_user_reviews.csv")
    project_root = os.path.abspath(os.path.join(os.getcwd()))
    review_data=pd.read_csv(os.path.join(project_root, args.csv_save_dir))
    review_data_sample=review_data.dropna()
    review_data_sample=review_data_sample.iloc[:args.row_number,] 
    review_data_sample['Rating Interval'] = review_data_sample['Sentiment'].apply(categorize_Sentiment)
    review_data_sample['Translated_Review'] = review_data_sample['Translated_Review'].apply(remove_punc_stopword)
    review_data_sample['Translated_Review'] = review_data_sample['Translated_Review'].str.join(' ')
    for i in review_data_sample.index:
        if len(review_data_sample['Translated_Review'][i])<=args.sentence_len:
            review_data_sample=review_data_sample.drop(i)
    Review = review_data_sample["Translated_Review"]
    Rating = review_data_sample["Rating Interval"]
    train_comments, test_comments, train_labels, test_labels  = train_test_split(list(Review), list(Rating), test_size = 0.2, random_state = 68)
    #------------------------------------------------------------------------

    net = BERTClassifier(output_dim=args.output_dim)            # BERTClassifier分类器，因为最终结果为3分类，所以输出维度为3，代表概率分布
    net = net.to(device)                                        # 将模型存放到GPU中，加速计算
    # 定义tokenizer对象，用于将评论语句转化为BertModel的输入信息
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    loss = nn.CrossEntropyLoss()                                # 损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)   # 小批量随机梯度下降算法
    max_acc = 0.3  # 初始化模型最大精度为0.5
    # 先测试未训练前的模型精确度
    train_acc = evaluate(net, args.batch_size, tokenizer, train_comments, train_labels)
    test_acc = evaluate(net, args.batch_size, tokenizer, test_comments, test_labels)
    Train_acc=[]
    Test_acc=[]

    # 输出精度
    print('--epoch', 0, '\t--train_acc:', train_acc, '\t--test_acc', test_acc)
    for epoch in tqdm(range(args.epochs)):
        i, sum_loss = 0, 0  # 每次开始训练时， i 为 0 表示从第一条数据开始训练
        # 开始训练模型
        while i < len(train_comments):
            comments = train_comments[i: min(i + args.batch_size, len(train_comments))]  # 批量训练，每次训练8条样本数据
            # 通过 tokenizer 数据化输入的评论语句信息，准备输入bert分类器
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(args.device)
            # 将数据输入到bert分类器模型中，获得结果
            res = net(tokens_X)
            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i + args.batch_size, len(train_comments))]).reshape(-1).to(args.device)
            optimizer.zero_grad()  # 清空梯度
            l = loss(res, y)  # 计算损失
            l.backward()  # 后向传播
            optimizer.step()  # 更新梯度
            sum_loss += l.detach()  # 累加损失
            i += args.batch_size  # 样本下标累加
        # 计算训练集与测试集的精度
        train_acc = evaluate(net, args.batch_size, tokenizer,  train_comments, train_labels)
        test_acc = evaluate(net, args.batch_size, tokenizer, test_comments, test_labels)
        Train_acc.append(train_acc.cpu().numpy())
        Test_acc.append(test_acc.cpu().numpy())
        # 输出精度
        print('\n--epoch', epoch+1, '\t--loss:', sum_loss / (len(train_comments) / args.batch_size), '\t--train_acc:', train_acc,
              '\t--test_acc', test_acc)
        # 如果测试集精度 大于 之前保存的最大精度，保存模型参数，并重设最大值
        if test_acc > max_acc:
            # 更新历史最大精确度
            max_acc = test_acc
            # 保存模型
            torch.save(net.state_dict(), 'bert.parameters')

 
    plt.plot(np.linspace(0,15,15), Train_acc, "ro-", label="train acc") 
    plt.plot(np.linspace(0,15,15), Test_acc, "bs-", label="test acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title("Bert")
    plt.show()

if __name__ == '__main__':  ### 判断当前的运行环境是否是直接运行的主程序
    parser = argparse.ArgumentParser() #作用是用于解析命令行参数
    # data param
    parser.add_argument('--csv_save_dir', type=str, help="dir to csv for training",default=os.path.join("ESE527", "googleplaystore_user_reviews.csv"))
    # model param
    parser.add_argument('--row_number', type=int, default=10000)
    parser.add_argument('--sentence_len', type=float, default=30)
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--output_dim', type=int, default=3)   #输出类别
    parser.add_argument('--batch_size', type=int, default=8) #训练批次
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    opt = parser.parse_args()
    main(opt)

