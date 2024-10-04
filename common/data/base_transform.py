import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def create_base_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """
    num_segments = args.num_segments if 'num_segments' in args else 1
    additional_targets = {}
    # for i in range(1, num_segments):
    #     additional_targets[f'image{i}'] = 'image'
    if split == 'train':
        base_transform = alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            alb.HorizontalFlip(),
            alb.augmentations.transforms.ToGray(p=0.01),
            alb.Resize(args.image_size, args.image_size),
            # alb.RandomResizedCrop(args.image_size, args.image_size,scale=(0.2, 1), p=1),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    elif split == 'test':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    return base_transform



def create_base_sbi_transforms(args, split='train'):
    """Base data transformation

    Args:
        args: Data transformation args
        split (str, optional): Defaults to 'train'.

    Returns:
        [transform]: Data transform
    """

    if split == 'train':
        base_transform = alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
			ToTensorV2(),
		], 
		additional_targets={f'image1': 'image'},
		p=1.)

    elif split == 'val':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])

    elif split == 'test':
        base_transform = alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])

    return base_transform
