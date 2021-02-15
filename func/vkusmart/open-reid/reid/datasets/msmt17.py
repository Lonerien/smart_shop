from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class MSMT17(Dataset):
#     url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'
#     md5 = '65005ab7d12ec1c44de4eeafe813e68a'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(MSMT17, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)  # 2373

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
#         mkdir_if_missing(raw_dir)

#         # Download the raw zip file
#         fpath = osp.join(raw_dir, 'Market-1501-v15.09.15.zip')
#         if osp.isfile(fpath) and \
#           hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
#             print("Using downloaded file: " + fpath)
#         else:
#             raise RuntimeError("Please download the dataset manually from {} "
#                                "to {}".format(self.url, fpath))

#         # Extract the file
        exdir = osp.join(raw_dir, 'MSMT17_V2')
#         if not osp.isdir(exdir):
#             print("Extracting zip file")
#             with ZipFile(fpath) as z:
#                 z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 4,101 identities with 15 camera views each
        identities = [[[] for _ in range(15)] for _ in range(4101)]

        def register(filename, pattern=re.compile(r'([\d]+)_([\d]+)_(\d\d)')):
#             fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            with open(osp.join(exdir, filename + '.txt'), 'r+') as list_txt:
#                 fpaths = sorted(list_txt.readlines())
                pids = set()
                for fpath in list_txt:
                    fname = osp.basename(fpath)
                    pid, _, cam = map(int, pattern.search(fname).groups())
                    assert 0 <= pid <= 4100
                    assert 1 <= cam <= 15
                    cam -= 1
                    pids.add(pid if filename == 'list_trainval' else pid + 1041)  # !!!
                    fname = ('{:08d}_{:02d}_{:04d}.jpg'
                             .format(pid, cam, len(identities[pid][cam])))
                    identities[pid][cam].append(fname)
                    fpath_dir = osp.join(exdir, 'mask_train_v2' if filename == 'list_trainval' else 'mask_test_v2')
                    shutil.copy(osp.join(fpath_dir, fpath.split()[0].strip()), osp.join(images_dir, fname))
            return pids

        # train + val == mask_train_v2 (almost)
        # query + gallery == mask_test_v2 (almost)
        # mask_test_v2 = 99939 images
        # list_gallery == 82161 images
        # list_query == 11659 images
        # mask_train_v2 == 34702 images
        # list_train == 30248 images
        # list_val == 2373 images
        trainval_pids = register('list_trainval')
        gallery_pids = register('list_gallery')
        query_pids = register('list_query')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'MSMT17', 'shot': 'multiple', 'num_cameras': 15,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
