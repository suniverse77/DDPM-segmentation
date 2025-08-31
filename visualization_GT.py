import numpy as np
from PIL import Image
import os

class MakePalette():
    def __init__(self):
        ffhq_palette = [
            1.0000,  1.0000 , 1.0000,
            0.4420,  0.5100 , 0.4234,
            0.8562,  0.9537 , 0.3188,
            0.2405,  0.4699 , 0.9918,
            0.8434,  0.9329  ,0.7544,
            0.3748,  0.7917 , 0.3256,
            0.0190,  0.4943 , 0.3782,
            0.7461 , 0.0137 , 0.5684,
            0.1644,  0.2402 , 0.7324,
            0.0200 , 0.4379 , 0.4100,
            0.5853 , 0.8880 , 0.6137,
            0.7991 , 0.9132 , 0.9720,
            0.6816 , 0.6237  ,0.8562,
            0.9981 , 0.4692 , 0.3849,
            0.5351 , 0.8242 , 0.2731,
            0.1747 , 0.3626 , 0.8345,
            0.5323 , 0.6668 , 0.4922,
            0.2122 , 0.3483 , 0.4707,
            0.6844,  0.1238 , 0.1452,
            0.3882 , 0.4664 , 0.1003,
            0.2296,  0.0401 , 0.3030,
            0.5751 , 0.5467 , 0.9835,
            0.1308 , 0.9628,  0.0777,
            0.2849  ,0.1846 , 0.2625,
            0.9764 , 0.9420 , 0.6628,
            0.3893 , 0.4456 , 0.6433,
            0.8705 , 0.3957 , 0.0963,
            0.6117 , 0.9702 , 0.0247,
            0.3668 , 0.6694 , 0.3117,
            0.6451 , 0.7302,  0.9542,
            0.6171 , 0.1097,  0.9053,
            0.3377 , 0.4950,  0.7284,
            0.1655,  0.9254,  0.6557,
            0.9450  ,0.6721,  0.6162
        ]
        self.ffhq_palette = [int(item * 255) for item in ffhq_palette]
        
        self.bedroom_palette = [
            255,  255,  255, # bg
            238,  229,  102, # bed
            255, 72, 69,     # bed footboard
            124,  99 , 34,   # bed headboard
            193 , 127,  15,  # bed side rail
            106,  177,  21,  # carpet
            248  ,213 , 43,  # ceiling
            252 , 155,  83,  # chandelier / ceiling fan blade
            220  ,147 , 77,  # curtain
            99 , 83  , 3,    # cushion
            116 , 116 , 138, # floor
            63  ,182 , 24,   # table/nightstand/dresser
            200  ,226 , 37,  # table/nightstand/dresser top
            225 , 184 , 161, # picture / mirrow
            233 ,  5  ,219,  # pillow
            142 , 172  ,248, # lamp column
            153 , 112 , 146, # lamp shade
            38  ,112 , 254,  # wall
            229 , 30  ,141,  # window
            99, 205, 255,    # curtain rod
            74, 59, 83,      # window frame
            186, 9, 0,       # chair
            107, 121, 0,     # picture / mirrow frame
            0, 194, 160,     # plinth
            255, 170, 146,   # door / door frame
            255, 144, 201,   # pouf
            185, 3, 170,     # wardrobe
            221, 239, 255,   # plant
            0, 0, 53,        # table staff
        ]

        self.cat_palette = [
            255,  255,  255,
            190, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            102, 102, 156,
            152, 251, 152,
            119, 11, 32,
            244, 35, 232,
            220, 20, 60,
            52 , 83  ,84,
            194 , 87 , 125,
            143 , 176 , 255,
            31 , 102 , 211,
            104 , 131 , 101
        ]

        self.horse_palette = [
            255,  255,  255,
            255, 74, 70,
            0, 137, 65,
            0, 111, 166,
            163, 0, 89,
            255, 219, 229,
            122, 73, 0,
            0, 0, 166,
            99, 255, 172,
            183, 151, 98,
            0, 77, 67, 
            143, 176, 255,
            241, 38, 110, 
            27, 210, 105,
            128, 150, 147,
            228, 230, 158,
            160, 136, 106,
            79, 198, 1,
            59, 93, 255,
            115, 214, 209,
            255, 47, 128
        ]

        self.celeba_palette = [
            255,  255,  255, # 0 background
            238,  229,  102,# 1 cloth
            250, 150,  50,# 2 ear_r
            124,  99 , 34, # 3 eye_g
            193 , 127,  15,# 4 hair
            225,  96  ,18, # 5 hat
            220  ,147 , 77, # 6 l_brow
            99 , 83  , 3, # 7 l_ear
            116 , 116 , 138,  # 8 l_eye
            200  ,226 , 37, # 9 l_lip
            225 , 184 , 161, # 10 mouth
            142 , 172  ,248, # 11 neck
            153 , 112 , 146, # 12 neck_l
            38  ,112 , 254, # 13 nose
            229 , 30  ,141, # 14 r_brow
            52 , 83  ,84, # 15 r_ear
            194 , 87 , 125, # 16 r_eye
            248  ,213 , 42, # 17 skin
            31 , 102 , 211, # 18 u_lip
        ]

    def ffhq(self):
        return self.ffhq_palette
    
    def bedroom(self):
        return self.bedroom_palette

    def cat(self):
        return self.cat_palette
    
    def horse(self):
        return self.horse_palette

    def celeba(self):
        return self.celeba_palette
    
def colorize_mask(mask, color):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(color)
    mask_map = np.array(new_mask.convert('RGB'))

    return mask_map

def load_directory(dataset_name):
    if dataset_name == 'bedroom':
        return './datasets/bedroom_28/real/test'
    if dataset_name == 'ffhq':
        return './datasets/ffhq_34/real/test'
    if dataset_name == 'celeba':
        return './datasets/celeba_19/real/test'
    if dataset_name == 'cat':
        return './datasets/cat_15/real/test'
    if dataset_name == 'horse':
        return './datasets/horse_21/real/test'

if __name__ == '__main__':
    # bedroom, ffhq, celeba, cat, horse
    dataset_name = 'cat'
    directory = load_directory(dataset_name)
    file_names = ['cat_0000001', 'cat_0000009', 'cat_0000012']

    paths = [os.path.join(directory, file + '.npy') for file in file_names]
    #'/home/suno3534/data/Label_Efficient/datasets/cat_15/real/test/cat_0000001.npy'
    palette = MakePalette()
    color = getattr(palette, dataset_name)()

    result_path = os.path.join('visualization', dataset_name)
    os.makedirs(result_path, exist_ok=True)

    for path, file in zip(paths, file_names):
        np_data = np.load(path).astype(np.uint8)
        mask_map = colorize_mask(np_data, color)
        save_path = os.path.join(result_path, file + '.png')
        Image.fromarray(mask_map).save(save_path)
