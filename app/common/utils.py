import jieba
from pathlib import Path

# 创建或加载停用词列表（可选）
stopwords_file = '../hit_stopwords.txt'  
if Path(stopwords_file).exists():
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
else:
    stopwords = set()


def tokenize(text):
    # 使用 jieba 分词，并去除停用词
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords and len(word.strip()) > 0]
    return " ".join(filtered_words)  # 将分词结果合并成字符串