import os
import random
import shutil

random.seed(12345)

def mkd(path):
    if not os.path.exists(path):
        os.mkdir(path)

base_dir = r'data\WHU_CD\split_file'

for p1 in ['train','test','val']:
    mkd(os.path.join(base_dir, p1))
    for p2 in ['A','B','label']:
        mkd(os.path.join(base_dir, p1, p2))

file_list = os.listdir(os.path.join(base_dir,'source','A'))
file_list = [name.split('before_')[1:] for name in file_list]
random.shuffle(file_list)
print(file_list)

for n1,n2,t in [(0,6096,'train'),(6096,6096+762,'test'),(6096+762,6096+762+762,'val')]:
    for i in range(n1,n2):
        for t1, t2 in [('A', 'before_'), ('B', 'after_'), ('label', 'change_label_')]:
            source_path = os.path.join(base_dir, 'source', t1, t2 + file_list[i][0])
            target_path = os.path.join(base_dir, t, t1, t + '_' + file_list[i][0])
            shutil.move(source_path, target_path)