import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


# class InterpolationMode(Enum):
#     """Interpolation modes
#     Available interpolation methods are ``nearest``, ``nearest-exact``, ``bilinear``, ``bicubic``, ``box``, ``hamming``,
#     and ``lanczos``.
#     """

#     NEAREST = "nearest"
#     NEAREST_EXACT = "nearest-exact"
#     BILINEAR = "bilinear"
#     BICUBIC = "bicubic"
#     # For PIL compatibility
#     BOX = "box"
#     HAMMING = "hamming"
#     LANCZOS = "lanczos"


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        # crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        auto_augment_policy=None,
        hflip_prob=0.5,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        viz_mode=False,
    ):

        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.hflip_prob = hflip_prob
        self.auto_augment_policy = auto_augment_policy
        self.ra_magnitude = ra_magnitude
        self.augmix_severity = augmix_severity
        self.random_erase_prob = random_erase_prob
        self.backend = backend
        self.viz_mode = viz_mode
        # self.transforms = transforms.Compose(trans)

    def get_list_train(self):
        # trans.append(transforms.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        trans = []
        backend = self.backend.lower()
        if backend == "tensor":
            trans.append(transforms.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        if self.hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(self.hflip_prob))
        if self.auto_augment_policy is not None:
            if self.auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=self.interpolation, magnitude=self.ra_magnitude))
            elif self.auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=self.interpolation))
            elif self.auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=self.interpolation, severity=self.augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(self.auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=self.interpolation))

        if backend == "pil" and self.viz_mode is False:
            trans.append(transforms.PILToTensor())

        if self.viz_mode is False:
            trans.extend(
                [
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )

        if self.random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=self.random_erase_prob))


    def get_list_test(self):
        trans = []
        backend = self.backend.lower()
        if backend == "tensor":
            trans.append(transforms.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        trans += [
            transforms.Resize(self.resize_size, interpolation=self.interpolation, antialias=True),
            # transforms.CenterCrop(self.crop_size),
        ]

        if backend == "pil":
            trans.append(transforms.PILToTensor())

        trans += [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return trans