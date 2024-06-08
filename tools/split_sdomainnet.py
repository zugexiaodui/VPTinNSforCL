import os
import os.path as osp

root_dir = f"DATA_ROOT_DIR"

def split_func(mode):
    mode_dir = osp.join(root_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    with open(f"./sdomain_{mode}.txt") as f:
        lines = f.readlines()
    for line in lines:
        rel_path, label = line.strip('\n').split(' ')
        src_path = osp.realpath(osp.join(root_dir, rel_path))
        assert osp.exists(src_path)
        class_dir = osp.realpath(osp.join(mode_dir, label.zfill(3)))
        if not osp.exists(class_dir):
            os.mkdir(class_dir)
        file_name = osp.basename(src_path)
        dst_path = osp.join(class_dir, file_name)
        os.symlink(src_path, dst_path)

split_func('train')
split_func('val')