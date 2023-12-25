import gensim
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter

# 日本語のテキストを取得するなどして、コーパスを用意
# ここでは簡単な例としてリストに日本語のテキストを入れています
japanese_corpus = [
    '王女 男',
    '王子 女',
    '皇帝 妃',
    '王女 女',  # '王女'を追加しました
]

# Janomeを使って形態素解析を行う
tokenizer = Tokenizer()
analyzer = Analyzer(tokenizer=tokenizer, token_filters=[POSKeepFilter(['名詞'])])

# 形態素解析をして単語のリストを得る
tokenized_corpus = [list(token.surface for token in analyzer.analyze(text)) for text in japanese_corpus]

# Word2Vecモデルを学習
model = Word2Vec(sentences=tokenized_corpus,
                 vector_size=200, window=5,
                 epochs=10, min_count=1, sg=0)

# ボキャブラリを確認
print(model.wv.key_to_index.keys())

# 類似した単語を求める
result = model.wv.most_similar(positive=['王女', '男'], negative=['女'])
print(result)