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
