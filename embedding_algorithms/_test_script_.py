from embedding_algorithms import *


data = [
    'I have a dream',
    'I have a apple',
    'you have a dream',
    'you have a apple',
    'he have a dream',
    'she have a apple',
]
new_data = []
for d in data:
    new_data.append(d.split(' '))

# m = Word2VecEmbedding(new_data, None, vec_dim=10, epoch=1000)
# vocab, vec = m.generate_embedding()
#
# m = Doc2VecEmbedding(new_data, None, vec_dim=10, epoch=1000)
# vocab, vec = m.generate_embedding()


m = FastEmbedding('./input.txt', new_data, None, vec_dim=10, epoch=10)
m.generate_embedding()