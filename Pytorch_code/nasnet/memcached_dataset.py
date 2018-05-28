import mc
from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img
 
class McDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s"%meta_file)
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        print("read meta done")
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        ## memcached
        server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
        mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
        value = mc.pyvector()
        mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        img = pil_loader(value_str)
        #img = np.zeros((350, 350, 3), dtype=np.uint8)
        #img = Image.fromarray(img)
        #cls = 0
        
        ## transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
