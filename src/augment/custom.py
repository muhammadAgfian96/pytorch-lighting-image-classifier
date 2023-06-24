import albumentations as al
from albumentations.pytorch.transforms import ToTensorV2

class CustomAugmentation:
    def __init__(self, 
                 input_size , 
                 mean=(0.485, 0.456, 0.406), 
                 std=(0.229, 0.224, 0.225), 
                 viz_mode=False) -> None:
        self._p_low = 0.25
        self._p_default = 0.5
        self._p_medium = 0.75
        self._p_highest = 0.95
        self.mean = mean
        self.std = std
        self.input_size = input_size
        self.viz_mode = viz_mode
        
    def _required_tail(self):
        ls_req = [
            al.Resize(height=self.input_size, width=self.input_size, always_apply=True),
            al.Normalize(mean=self.mean, std=self.std, always_apply=True, max_pixel_value=255.0),
            ToTensorV2(always_apply=True),
        ]
        if self.viz_mode:
            return [ls_req[0]]
        return ls_req

    def get_list_train(self):
        common = [
            al.OneOf(p=1.0, transforms=[
                al.VerticalFlip(p=self._p_highest),
                al.HorizontalFlip(p=self._p_highest),
            ]),
            al.ShiftScaleRotate(
                always_apply=True,
                shift_limit= [-0.12, 0.12],
                scale_limit= [-0.05, 0.15],
                rotate_limit= [-45, 45],
                interpolation= 0,
                border_mode= 0,
                value= [0, 0, 0]
            )
        ]

        coarse = [
            al.CoarseDropout(
                always_apply=False, p=self._p_low,
                min_holes=16, max_holes=25, 
                min_height=0.02, max_height=0.12, 
                min_width=0.02, max_width=0.12,
            )
        ]

        brightness = [
            al.RandomBrightnessContrast(
                p=self._p_medium, 
                brightness_limit=[-0.25, 0.25],
                contrast_limit=[-0.25, 0.25],
                brightness_by_max=False
            )
        ]

        blur = [
            al.OneOf(p=self._p_low, transforms=[
                    al.MotionBlur(p=1.0),
                    al.ImageCompression(p=1.0),
                    al.OpticalDistortion(p=1.0),
                    al.MultiplicativeNoise(p=1.0)
                ]
            ),
        ]

        noise = [
            al.OneOf(p=self._p_default, transforms=[
                    al.GaussNoise(p=1.0, var_limit=(101.97, 322.37), ),
                    al.ISONoise(p=1.0),
                    al.RandomGamma(p=1.0, gamma_limit=(57, 142), ),
                    # al.RandomFog(always_apply=False, p=1.0, fog_coef_lower=0.23, fog_coef_upper=0.27, alpha_coef=0.92),
                    al.RandomRain(always_apply=False, p=1.0, slant_lower=-10, slant_upper=12, drop_length=10, drop_width=1, drop_color=(248, 246, 247), blur_value=2, brightness_coefficient=0.84, rain_type=None),
                ]
            )
        ]

        tail = self._required_tail()

        ls_compose = []
        ls_compose.extend(common)
        ls_compose.extend(coarse)
        ls_compose.extend(brightness)
        ls_compose.extend(blur)
        ls_compose.extend(noise)
        ls_compose.extend(tail)
        return ls_compose

    def get_list_test(self):
        return self._required_tail()

