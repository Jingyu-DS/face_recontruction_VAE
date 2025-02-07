import os
import imageio
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoad:
    def __init__(self, data_path, attribute_path):
        self.data_path = data_path
        self.attribute_path = attribute_path
    
    
    def fetch_dataset(self, dx, dy, dimx, dimy):
        df_attrs = pd.read_csv(self.attribute_path, sep = "\t", skiprows = 1,)
        df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns = df_attrs.columns[1:])

        photo_ids = []
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for fname in filenames:
                if fname.endswith(".jpg"):
                    fpath = os.path.join(dirpath, fname)
                    photo_id = fname[:-4].replace('_',' ').split()
                    person_id = ' '.join(photo_id[:-1])
                    photo_number = int(photo_id[-1])
                    photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})
        
        photo_ids = pd.DataFrame(photo_ids)
        df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))
        assert len(df)==len(df_attrs),"lost some data when merging dataframes"

        all_photos = df['photo_path'].apply(imageio.imread)\
                                .apply(lambda img:img[dy:-dy,dx:-dx])\
                                .apply(lambda img: np.array(Image.fromarray(img).resize([dimx,dimy])) )

        all_photos = np.stack(all_photos.values).astype('uint8')
        all_attrs = df.drop(["photo_path","person","imagenum"],axis=1)

        return all_photos, all_attrs


"""
Example Usage:
dataLoader = DataLoad(DATASET_PATH, ATTRIBUTES_PATH)
data, attrs = dataLoader.fetch_dataset(dx, dy, dimx, dimy)
data = np.array(data / 255, dtype='float32')
"""
