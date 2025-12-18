from flemme.dataset import create_loader
from flemme.utils import load_config
from datasets import Dataset
from tqdm import tqdm
import torch
### medpoints-cls
# loader_config = load_config("./cls_loader.yaml")['loader']
# loader = create_loader(loader_config)
# data_loader = loader['data_loader']
# print('loading dataset ...')
# input_label_list = [(t[0], t[1]) for t in tqdm(data_loader, desc="loading")]

# data_list = list(torch.cat([ pt[0] for pt in input_label_list], dim = 0).cpu().numpy())
# label_list = list(torch.cat([ pt[1] for pt in input_label_list], dim = 0).cpu().numpy())

# dict = {
#     "data": data_list,
#     "label": label_list
# }
# hf_dataset = Dataset.from_dict(dict)
# print('pushing dataset to hugging face ...')
# hf_dataset.push_to_hub("wlsdzyzl/MedPointS-cls")

# ### medpoints-cpl 2048
# loader_config = load_config("./cpl_loader.yaml")['loader']
# loader = create_loader(loader_config)
# data_loader = loader['data_loader']
# print('loading dataset ...')
# partial_target_label_list = [(t[0], t[1], t[2]) for t in tqdm(data_loader, desc="loading")]

# partial_list = list(torch.cat([ pt[0] for pt in partial_target_label_list], dim = 0).cpu().numpy())
# target_list = list(torch.cat([ pt[1] for pt in partial_target_label_list], dim = 0).cpu().numpy())
# label_list = list(torch.cat([ pt[2] for pt in partial_target_label_list], dim = 0).cpu().numpy())

# dict = {
#     "partial": partial_list,
#     "target": target_list,
#     "label": label_list
# }
# hf_dataset = Dataset.from_dict(dict)
# print('pushing dataset to hugging face ...')
# hf_dataset.push_to_hub("wlsdzyzl/MedPointS-cpl")

### medpoints-seg
loader_config = load_config("./seg_loader.yaml")['loader']
loader = create_loader(loader_config)
data_loader = loader['data_loader']
print('loading dataset ...')
input_label_list = [(t[0], t[1]) for t in tqdm(data_loader, desc="loading")]

data_list = list(torch.cat([ pt[0] for pt in input_label_list], dim = 0).cpu().numpy())
label_list = list(torch.cat([ pt[1] for pt in input_label_list], dim = 0).cpu().numpy())

dict = {
    "partial": data_list,
    "label": label_list
}
hf_dataset = Dataset.from_dict(dict)
print('pushing dataset to hugging face ...')
hf_dataset.push_to_hub("wlsdzyzl/MedPointS-seg")