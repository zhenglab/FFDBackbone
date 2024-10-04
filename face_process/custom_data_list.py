import os


def get_image_files(directory):
    image_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(os.path.join(root, file))

    return image_files


def get_list(root=None):
    #Celeb-v
    root = '/HDD0/guozonghui/project/datasets/celeb-v/frames'
    celeb_v = get_image_files(root)
    print('celeb-v:', len(celeb_v))
    #FFHQ
    root = '/SSD2/shiliang/work/data/FFHQ_256'
    ffhq = get_image_files(root)
    print('celeb-v:', len(ffhq))

if __name__=='__main__':
    get_list()