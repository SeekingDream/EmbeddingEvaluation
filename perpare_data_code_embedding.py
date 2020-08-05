import javalang
import os
from utils import parse_statement, parse_source




def main():
    err_num = 0
    dir_name = './dataset/raw_code/'
    out_dir = './dataset/code_embedding/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for project in os.listdir(dir_name):
        sub_dir = os.path.join(dir_name, project)
        sub_out = os.path.join(out_dir, project)
        if not os.path.isdir(sub_out):
            os.mkdir(sub_out)
        for file_name in os.listdir(sub_dir):
            try:
                with open(os.path.join(sub_dir, file_name), 'r') as f:
                    source_code = f.readlines()
                new_source_code = parse_source(source_code)
                with open(os.path.join(sub_out, file_name), 'w') as f:
                    f.writelines(new_source_code)
                    print('finish', project, file_name)
            except:
                err_num += 1
                print('error', project, file_name)
                pass
    print(err_num)

if __name__ == '__main__':
    main()
