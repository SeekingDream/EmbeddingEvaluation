import numpy as np
import torch
import random
import javalang


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_statement(code):
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        tokens = [tk.value for tk in tokens]
        code = " ".join(tokens)
    except:
        pass
    return code


def parse_source(source_code):
    for i, code in enumerate(source_code):
        code = parse_statement(code)
        source_code[i] = code
    return source_code


#
# def configure_exp(embed_type, embed_path):
#     train_path = args.train_data
#     test_path = args.test_data
#     pre_embedding_path = args.embedding_path
#     if args.embedding_type == 0:
#         d_word_index, embed = torch.load(pre_embedding_path)
#         print('load existing embedding vectors, name is ', pre_embedding_path)
#     elif args.embedding_type == 1:
#         v_builder = VocabBuilder(path_file=train_path)
#         d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
#         print('create new embedding vectors, training from scratch')
#     elif args.embedding_type == 2:
#         v_builder = VocabBuilder(path_file=train_path)
#         d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
#         embed = torch.randn([len(d_word_index), args.embedding_dim]).cuda()
#         print('create new embedding vectors, training the random vectors')
#     else:
#         raise ValueError('unsupported type')
#     if embed is not None:
#         if type(embed) is np.ndarray:
#             embed = torch.tensor(embed, dtype=torch.float).cuda()
#         assert embed.size()[1] == args.embedding_dim