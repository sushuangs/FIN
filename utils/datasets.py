import os
import glob
from PIL import Image
from natsort import natsorted
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import config as c

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class Test_Dataset(Dataset):
    def __init__(self, path, format):
        self.transform = T.Compose([
            T.Resize(128),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.files = natsorted(sorted(glob.glob(path + "/*." + format)))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


# class INN_Dataset(Dataset):
#     def __init__(self, transforms, mode="train"):
#         self.transform = transforms
#         self.mode = mode
#         self.image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp', 'ppm', 'pgm')  # 支持多种图像格式

#         # 确定基础路径
#         base_path = c.TRAIN_PATH if mode == 'train' else c.VAL_PATH

#         # 递归获取所有匹配的图像文件
#         self.files = []
#         for ext in self.image_extensions:
#             pattern = os.path.join(base_path, '**', f'*.{ext}')
#             self.files.extend(glob(pattern, recursive=True))
        
#         # 自然排序确保文件顺序一致性
#         self.files = natsorted(self.files)

#     def __getitem__(self, index):
#         # 处理索引越界的情况
#         if index >= len(self.files):
#             raise IndexError(f"Index {index} out of range for dataset with size {len(self.files)}")
        
#         try:
#             with Image.open(self.files[index]) as image:
#                 image = to_rgb(image)  # 确保转换为RGB格式
#                 return self.transform(image)
#         except Exception as e:
#             print(f"Error loading {self.files[index]}: {str(e)}")
#             # 自动跳过错误文件，尝试下一个索引
#             return self.__getitem__((index + 1) % len(self.files))

#     def __len__(self):
#         return len(self.files)

class INN_Dataset(Dataset):
    def __init__(self, transforms, mode="train", extensions=('jpg', 'jpeg', 'png', 'bmp')):
        """
        参数:
            root (str): 数据集根目录（需包含按类别划分的子文件夹）
            H, W (int): 最终输出的图像尺寸
            extensions (tuple): 支持的图像文件扩展名
        """
        super().__init__()
        self.root = c.TRAIN_PATH if mode == 'train' else c.VAL_PATH
        self.extensions = {'.' + ext.lstrip('.').lower() for ext in extensions}
        
        # 初始化类别映射和样本列表
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

        self.transform = transforms
    
    def _find_classes(self):
        """获取类别名称到索引的映射"""
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        return classes, {cls_name: i for i, cls_name in enumerate(classes)}
    
    def _is_valid_image(self, path):
        """检查是否为有效图像（尺寸、比例、文件格式）"""
        try:
            with Image.open(path) as img:
                # 检查文件格式
                if os.path.splitext(path)[1].lower() not in self.extensions:
                    return False
                
                # 检查宽高比例是否合理
                ratio = img.width / img.height
                if ratio < 0.5 or ratio > 2.0:
                    return False
                
                return True
        except (IOError, OSError):
            return False
    
    def _make_dataset(self):
        """构建有效样本列表（预处理阶段过滤）"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                if os.path.isfile(path) and self._is_valid_image(path):
                    samples.append((path, class_idx))
        
        return samples
    
    def __getitem__(self, index):
        """直接加载预验证过的样本"""
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            # 如果出现意外错误，返回随机样本避免中断训练
            return self[torch.randint(0, len(self), (1,)).item()]
    
    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self):
        """辅助方法：获取类别分布统计"""
        from collections import Counter
        return Counter([label for _, label in self.samples])

transform = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.Resize([c.cropsize, c.cropsize]),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_val = T.Compose([
    T.Resize([c.cropsize_val, c.cropsize_val]),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

trainloader = DataLoader(
    INN_Dataset(transforms=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
# Test Dataset loader
testloader = DataLoader(
    INN_Dataset(transforms=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)





