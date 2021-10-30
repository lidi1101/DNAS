from .derain import Derain_datasets
from .denoise import Denoise_datasets, Denoise_SIM_noise1800, Denoise_CBD_real

tasks_dict = {
    'derain': Derain_datasets,
    'denoise': Denoise_datasets,
    'denoise_SIM_noise1800': Denoise_SIM_noise1800,
    'denoise_CBD_real': Denoise_CBD_real
}


