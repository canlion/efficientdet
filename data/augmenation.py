import albumentations as A
import cv2


def get_agumentator(train=True, img_size=(512, 512), min_area=0., min_visibility=0.):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    transform = [
        A.LongestMaxSize(max_size=max(img_size), always_apply=True),
        A.PadIfNeeded(min_height=img_size[0], min_width=img_size[1],
                      border_mode=0, always_apply=True, value=[0] * 3)
    ]

    if train:
        transform.extend([
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=.3),
            A.RandomResizedCrop(height=img_size[0], width=img_size[1], p=.3),
            A.HorizontalFlip(p=.5),
            A.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.3),
            A.RGBShift(r_shift_limit=15, b_shift_limit=15, g_shift_limit=15)
        ])

    box_params = A.BboxParams(format='pascal_voc',
                              min_area=min_area,
                              min_visibility=min_visibility,
                              label_fields=['labels'])

    return A.Compose(transform, bbox_params=box_params)
