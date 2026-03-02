import torch
from torch.utils.data import Dataset

from common.misc import default_collate
from data.build import DATASETWRAPPER_REGISTRY
from data.data_utils import pad_tensor, pad_tensors


@DATASETWRAPPER_REGISTRY.register()
class LeoObjPadDatasetWrapper2(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.dataset_name = dataset.__class__.__name__
        # self.padding_keys = ['obj_pcds', 'obj_fts_img', 'obj_fts_vox', 'prompt', 'prompt_pad_masks']
        # self.not_padding_keys = ['scene_pcds', 'scene_3d_fts', 'scene_2d_fts', 'tgt_obj_mask']
        # self.padding_keys = ['scene_voxs', 'vox_3d_fts', 'vox_2d_fts', 'obj_fts_img', 'obj_fts_vox','obj_fts', 'prompt', 'prompt_pad_masks', 'agent_pos_ori', 'all_obj_imgs']
        self.padding_keys = ['obj_fts_img', 'obj_fts_vox','obj_fts', 'prompt', 'prompt_pad_masks', 'agent_pos_ori', 'all_obj_imgs', 'pred_obj_prob']
        self.not_padding_keys = ['index']
        self.max_thought_img_num = getattr(args, 'max_thought_img_num', 20)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, batch):
        new_batch = {}
        # pad to longest
        for k in self.padding_keys:
            if k in batch[0]:
                tensors = [sample.pop(k) for sample in batch]
                padding_value = -100 if k == 'obj_labels' else 0
                new_batch[k] = pad_tensors(tensors, pad=padding_value)

        # use tgt_obj_mask to generate vox_masks
        if 'tgt_obj_mask' in batch[0]:
            tensors = [sample.pop('tgt_obj_mask') for sample in batch]
            new_batch['tgt_obj_mask'], new_batch['vox_masks'] = pad_tensors(tensors, pad=0, return_mask=True)

        # use obj_ids to generate thought_img_masks
        tensors = [sample.pop('obj_ids') for sample in batch]
        new_batch['obj_ids'] = tensors
        _, new_batch['thought_img_masks'] = pad_tensors(tensors, pad=0, return_mask=True)

        # use obj_locs to generate obj_masks
        if 'obj_locs' in batch[0]:
            tensors = [sample.pop('obj_locs') for sample in batch]
            new_batch['obj_locs'], new_batch['obj_masks'] = pad_tensors(tensors, pad=0, return_mask=True)

        if 'grounding_obj_mask_gt' in batch[0]:
            tensors = [sample.pop('grounding_obj_mask_gt') for sample in batch]
            new_batch['grounding_obj_mask_gt'] = pad_tensors(tensors, pad=0)

        if 'pred_obj_prob' in batch[0]:
            tensors = [sample.pop('pred_obj_prob') for sample in batch]
            new_batch['pred_obj_prob'] = pad_tensors(tensors, pad=0.0)

        if 'obj_2d_fts' in batch[0]:
            tensors = [sample.pop('obj_2d_fts') for sample in batch]
            new_batch['obj_2d_fts'] = pad_tensors(tensors, pad=0)
            assert new_batch['obj_locs'].shape[1] == new_batch['obj_2d_fts'].shape[1]

        # not pad, collate in a list
        for k in self.not_padding_keys:
            if k in batch[0]:
                new_batch[k] = [sample.pop(k) for sample in batch]

        # default collate
        new_batch.update(default_collate(batch))
        return new_batch


@DATASETWRAPPER_REGISTRY.register()
class LeoObjPadDatasetWrapper(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.max_obj_len = args.max_obj_len
        self.dataset_name = dataset.__class__.__name__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]

        data_dict['obj_fts'] = pad_tensor(data_dict['obj_fts'], max_len=self.max_obj_len, pad=0.0).float()   # O, num_points, 6
        if 'obj_fts_img' in data_dict:
            data_dict['obj_fts_img'] = pad_tensor(data_dict['obj_fts_img'], max_len=self.max_obj_len, pad=0.0).float()   # O, D
        if 'obj_fts_vox' in data_dict:
            data_dict['obj_fts_vox'] = pad_tensor(data_dict['obj_fts_vox'], max_len=self.max_obj_len, pad=0.0).float()   # O, D
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))   # O
        data_dict['obj_locs'] = pad_tensor(data_dict['obj_locs'], max_len=self.max_obj_len, pad=0.0).float()   # O, 6

        return data_dict

    def collate_fn(self, batch):
        new_batch = {}
        # pad prompt to longest, for pq3d
        padding_keys = ['prompt', 'prompt_pad_masks']
        for k in padding_keys:
            if k in batch[0]:
                tensors = [sample.pop(k) for sample in batch]
                padding_value = -100 if k == 'obj_labels' else 0
                padded_tensor = pad_tensors(tensors, pad=padding_value)
                new_batch[k] = padded_tensor
        # default collate
        new_batch.update(default_collate(batch))

        return new_batch
