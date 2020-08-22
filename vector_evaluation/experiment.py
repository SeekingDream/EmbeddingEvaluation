import numpy as np
import torch
from gensim.models import word2vec
import datetime
import random
import warnings

warnings.filterwarnings('ignore')

from utils import set_random_seed
from Intuition.investigate_generability import constructModel, dict2list
from embedding_algorithms.word2vec import Word2VecEmbedding


def vocab_frequency(data_dir):
    sentence = word2vec.PathLineSentences(data_dir)
    model = word2vec.Word2Vec()
    model.build_vocab(sentences=sentence)
    torch.save(model.wv, 'wv_fre.fcy')


class Metric:
    def __init__(self):
        self.res = {
            'acc': 0.0,
            'map': 0.0,
            'ndcg': 0.0,
            'mrr': 0.0,
            'recall_3': 0.0
        }
        self.random = {
            'acc': 0.0,
            'map': 0.0,
            'ndcg': 0.0,
            'mrr': 0.0,
            'recall_3': 0.0
        }

    @staticmethod
    def MAP(ranked_list, ground_truth):
        hits = 0
        sum_precs = 0
        for n in range(len(ranked_list)):
            if ranked_list[n] in ground_truth:
                hits += 1
                sum_precs += hits / (n + 1.0)
        if hits > 0:
            return sum_precs / len(ground_truth)
        else:
            return 0

    @staticmethod
    def MRR(ranked_list: list, ground_truth):
        return 1 / (ranked_list.index(ground_truth[0]) + 1)

    @staticmethod
    def NDCG(rank_list, ground_truth):
        def getDCG(scores):
            return np.sum(
                np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                dtype=np.float32)
        relevance = np.ones_like(ground_truth)
        it2rel = {it: r for it, r in zip(ground_truth, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
        idcg = getDCG(np.sort(relevance)[::-1])
        dcg = getDCG(rank_scores)
        if dcg == 0.0:
            return 0.0
        ndcg = dcg / idcg
        return ndcg

    def update(self, preds, truth):
        self.res['acc'] += (preds[0] == truth[0])
        self.res['recall_3'] += (truth[0] in preds[:3])
        self.res['map'] += self.MAP(preds[:3], truth[:3])
        self.res['mrr'] += self.MRR(preds, truth)
        self.res['ndcg'] += self.NDCG(preds[:3], truth[:3])

        tmp = truth.copy()
        random.shuffle(tmp)
        self.random['acc'] += (tmp[0] == truth[0])
        self.random['recall_3'] += (truth[0] in tmp[:3])
        self.random['map'] += self.MAP(tmp[:3], truth[:3])
        self.random['mrr'] += self.MRR(tmp, truth)
        self.random['ndcg'] += self.NDCG(tmp[:3], truth[:3])

    def produce_final(self, exp_num):
        for k in self.res:
            self.res[k] = self.res[k] / exp_num
        for k in self.random:
            self.random[k] = self.random[k] / exp_num
        return self

    def print(self):
        print('Approach:')
        print(self.res)
        print('Random:')
        print(self.random)


class GroundTruth:
    res = [
        [69.48, 59.04, 59.04, 62.65, 59.44, 45.78, 47.38, 59.84, 18.48, 49.40],
        [25, 38, 32, 38, 21, 25, 20, 28, 37, 21],
        [53.68, 50.16, 61.9, 35.27, 62.03, 58.18, 5.07, 58.51, 56.32, 0.84],
        [27.46, 34.15, 29.8, 32.22, 29.3, 18.99, 30.65, 23.51, 19.07, 29.61],
        [63.85, 67.64, 70.03, 67.88, 69.27, 63.85, 66.3, 65.45, 63.85, 51.36]
    ]
    task = ['authorship', 'summary', 'search', 'completion', 'clone']
    embed = [
        'random',
        'Word2VecEmbedding0.vec',
        'Word2VecEmbedding1.vec',
        'FastEmbeddingcbow.vec',
        'FastEmbeddingskipgram.vec',
        'Doc2VecEmbedding0.vec',
        'Doc2VecEmbedding1.vec',
        'GloVeEmbeddingNone.vec',
        'ori_code2seq.vec',
        'code2vec.vec'
    ]
    task_num = 5
    embed_num = 10

    def get_tk_frec(self):
        mv = torch.load('wv_fre.fcy')
        vocab = mv.vocab
        frec = {}
        for k in self.word2index:
            new_k = ''.join(k.split('|')).lower()
            if new_k in vocab:
                frec[self.word2index[k]] = vocab[new_k].count
        sort_frec = sorted(frec.items(), key=lambda item: item[1])
        return frec, sort_frec

    def __init__(self, vec_dir, dim, thresh, top_fre):
        self.vec_dir = vec_dir
        self.res = [np.array(i).reshape([1, -1]) for i in self.res]
        self.truth = np.concatenate(self.res, axis=0)
        self.norm_score = self._normalize_truth(norm_type=0)
        self.thresh = thresh
        self.word2index, _ = torch.load(vec_dir + 'ori_code2seq.vec')
        self.index2word = dict2list(self.word2index)
        self.w2v_list = []
        self.vec_list = []
        self.frequency, self.sorted_fre = self.get_tk_frec()
        self.sorted_fre = self.sorted_fre[-top_fre:]

        self.token_num, self.vec_dim = len(self.word2index), dim
        for file_name in self.embed:
            if file_name == 'random':
                vec = np.random.randn(self.token_num, self.vec_dim)
            else:
                _, vec = torch.load(vec_dir + file_name)
            self.vec_list.append(vec)
            m = constructModel(vec, self.word2index, self.index2word)
            self.w2v_list.append(m)

    def _normalize_truth(self, norm_type):
        if norm_type == 0:
            max_acc, min_acc = np.max(self.truth, axis=1), np.min(self.truth, axis= 1)
            max_acc, min_acc = max_acc.reshape([-1, 1]), min_acc.reshape([-1, 1])
            norm_truth = (self.truth - min_acc) / (max_acc - min_acc + 1e-8)
        elif norm_type == 1:
            mean, std = np.mean(self.truth, axis=1), np.std(self.truth, axis=1)
            norm_truth = (self.truth - mean) / std
        else:
            raise ValueError
        score = np.average(norm_truth, axis=0)
        return score

    def calculate_score(self, base, candidate, sample_num):
        src_i = np.random.randint(0, len(self.sorted_fre), sample_num)
        src_i = [self.sorted_fre[i][0] for i in src_i]
        src_tk = [self.index2word[i] for i in src_i]
        tar_tk = []
        sim_mat = np.zeros([sample_num, 2 + len(candidate)])
        for i, s_tk in enumerate(src_tk):
            (t, s) = self.w2v_list[base].wv.most_similar(s_tk, topn=1)[0]
            tar_tk.append(t)
            sim_mat[i, 0] = s
            s_vec = self.vec_list[0][self.word2index[s_tk]]
            t_index = self.word2index[t]
            t_vec = self.vec_list[0][t_index:t_index + 1]
            sim = self.w2v_list[0].wv.cosine_similarities(s_vec, t_vec)[0]
            sim_mat[i, 1] = sim

        tar_i = [self.word2index[t] for t in tar_tk]
        for i, t_i in enumerate(tar_i):
            for j, c in enumerate(candidate):
                s_vec = self.vec_list[c][src_i[i]]
                t_vec = self.vec_list[c][t_i: t_i+1]
                sim = self.w2v_list[c].wv.cosine_similarities(s_vec, t_vec)[0]
                sim_mat[i, j + 2] = sim
        v = self.detector(sim_mat)
        return v / sample_num

    def detector(self, sim_mat):
        base_score = sim_mat[:, 0]
        rnd_score = sim_mat[:, 1]
        average = (base_score + rnd_score).reshape([-1, 1]) / 2
        cand_score = sim_mat[:, 2:]
        sampel_num, candiate_num = cand_score.shape
        # if det_type == 0:
        #     mean, std = np.mean(cand_score, axis=1), np.std(cand_score, axis=1)
        #     v = (base_score - mean) / (std + 1e-8)
        #     res = np.sum(abs(v) < self.thresh, dtype=np.float)
        # elif det_type == 1:
        res = np.sum(np.sum(cand_score > average, axis=1) > (candiate_num/2))
        return res

    def experiment(self, candidate_num, exp_num, sample_num):
        metric = Metric()
        st_time = datetime.datetime.now()
        for _ in range(exp_num):
            candidate = np.random.choice(
                 range(self.embed_num), size=candidate_num, replace=False)
            #candidate = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            candidate_score = np.zeros([candidate_num])
            truth_score = self.norm_score[candidate]
            for j, base in enumerate(candidate):
                index = [i for i in range(candidate_num) if i != j]
                baseline = candidate[index]
                s = self.calculate_score(base, baseline, sample_num)
                candidate_score[j] = s
            truth_sort = list(np.argsort(truth_score * -1))
            pred_sort = list(np.argsort(candidate_score * -1))
            metric.update(pred_sort, truth_sort)
        ed_time = datetime.datetime.now()
        time_cost = (ed_time - st_time) / (exp_num * candidate_num)
        metric.produce_final(exp_num)
        return metric, time_cost


def main():
    vec_dir = '/glusterfs/data/sxc180080/EmbeddingEvaluation/vec/100_2/'
    m = GroundTruth(vec_dir, dim=100, thresh=1, top_fre=50000)
    exp_num = 200
    sample_num = 1000
    for candidate_num in range(3, 10):
        metric, time_cost = m.experiment(candidate_num=candidate_num, exp_num=exp_num, sample_num=sample_num)
        print('candidate_num', candidate_num, 'time cost:', time_cost)
        metric.print()
        print('=====================================')


if __name__ == '__main__':
    a = [1, 2, 4]
    b = [2, 1, 3]
    Metric.NDCG(a, b)
    set_random_seed(100)
    main()
