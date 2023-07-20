'''
copy a diffusers deepspeed model state_dict to a composer deepspeed model.
'''
import os
import tarfile
import json
import tempfile
import torch

from diffusion.models.unet.unet_2d_condition import UNet2DConditionModel
from torchinfo import summary
import deepspeed
from collections import OrderedDict

_DEEPSPEED_TAG = 'deepspeed'  # always tag with the same, deterministic name. We'll rename the tarball to the appropriate name.

ds_ckpt = '/mnt/CV_teamz/users/zhongyi/workspace/checkpoint-2/pytorch_model/'

cp_ckpt = '/mnt/CV_teamz/users/zhongyi/workspace/diffusion-dgx/outputs/model_save_3B_deepspeed/deepspeed/'

new_cp_dir = '/mnt/CV_teamz/users/zhongyi/workspace/composer-2/deepspeed'
os.makedirs(new_cp_dir, exist_ok=True)


def get_model(unet_model_config_path,):
    unet_config = json.load(open(unet_model_config_path))
    unet = UNet2DConditionModel.from_config(unet_config)
    summary(unet)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=0.000001,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    deepspeed_config = {'zero_optimization': {'stage': 2}}
    unet, optimizer, _, = deepspeed.initialize(
        unet,
        optimizer,
        config=deepspeed_config,
    )
    unet.load_checkpoint(ds_ckpt)


def _get_write_mode(name: str) -> str:
    """Get the write mode to use with :func:`tarfile.open`."""
    if name.endswith('.tar'):
        return 'w'
    if name.endswith('.tar.gz') or name.endswith('.tgz'):
        return 'w:gz'
    if name.endswith('.tar.bz2'):
        return 'w:bz2'
    if name.endswith('.tar.lzma'):
        return 'w:xz'
    raise ValueError(f'{name} does not end with a valid tarfile extension.')


def _save_deepspeed_model(model, filename: str):
    """Save Deepspeed model and tarball the files."""
    write_mode = _get_write_mode(filename)
    read_mode = 'r' + write_mode[1:]

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_checkpoint(tmpdir, _DEEPSPEED_TAG)

        if os.path.exists(filename):
            # extract to tmpdir to append below
            # not all compression formats support direct append
            with tarfile.open(filename, read_mode) as tar:
                tar.extractall(tmpdir)

        with tarfile.open(filename, write_mode) as tar:
            tar.add(tmpdir, arcname='')


def df2cp(df_keys):
    unet_prefix = 'unet.'
    if 'tokenizer_adapter_layer.' in df_keys:
        return df_keys.replace('tokenizer_adapter_layer.', 'projs.clip_G.')
    if 'tokenizer_adapter_layer_clip.' in df_keys:
        return df_keys.replace('tokenizer_adapter_layer_clip.', 'projs.clip_B.')
    return unet_prefix + df_keys


def cp2df(cp_keys):
    unet_prefix = 'unet.'
    if 'projs.clip_G.' in cp_keys:
        return cp_keys.replace('projs.clip_G.', 'tokenizer_adapter_layer.')
    if 'projs.clip_B.' in cp_keys:
        return cp_keys.replace('projs.clip_B.', 'tokenizer_adapter_layer_clip.')
    if cp_keys.startswith(unet_prefix):
        return cp_keys.replace(unet_prefix, '')
    else:
        return None


def convert_model_states():

    state_name = 'mp_rank_00_model_states.pt'

    ds_state_path = os.path.join(ds_ckpt, state_name)
    ds_state_info = torch.load(ds_state_path, map_location='cpu')
    print('loaded diffusers deepspeed checkpoint')

    cp_state_path = os.path.join(cp_ckpt, state_name)
    cp_state_info = torch.load(cp_state_path, map_location='cpu')
    print('loaded composer deepspeed checkpoint')

    for key, cp_state_dict in cp_state_info.items():

        print(f'replace {key}')

        if key == 'module':

            for k in list(cp_state_dict.keys()):
                df_k = cp2df(k)
                if df_k is not None:
                    cp_state_dict[k] = ds_state_info[key][df_k]
                else:
                    if not k.startswith('text_encoder') and not k.startswith(
                            'vae'):
                        print(k)

        elif key == 'buffer_names':
            pass

        elif key == 'optimizer':
            pass

        elif key == 'param_shapes':

            for k in list(cp_state_dict[0].keys()):
                df_k = cp2df(k)
                if df_k is not None:
                    cp_state_dict[0][k] = ds_state_info[key][0][df_k]
                else:
                    if not k.startswith('text_encoder') and not k.startswith(
                            'vae'):
                        print(k)

        elif key == 'frozen_param_shapes':
            pass

        elif key == 'shared_params':
            pass

        elif key == 'frozen_param_fragments':
            pass

        elif key == 'lr_scheduler':
            pass

        elif key == 'data_sampler':
            pass

        elif key == 'random_ltd':
            pass

        elif key == 'sparse_tensor_module_names':
            pass

        elif key == 'skipped_steps':

            cp_state_info[key] = ds_state_info[key]

        elif key == 'global_steps':

            cp_state_info[key] = ds_state_info[key]

        elif key == 'global_samples':

            cp_state_info[key] = ds_state_info[key]

        elif key == 'dp_world_size':

            assert cp_state_info[key] == ds_state_info[key]

        elif key == 'mp_world_size':

            assert cp_state_info[key] == ds_state_info[key]

        elif key == 'ds_config':

            cp_state_info[key] = ds_state_info[key]

        elif key == 'ds_version':

            pass

        else:
            print(f'unconvert: {key}')
            __import__('ipdb').set_trace()
            print(f'unconvert: {cp_state_dict}')

    print(cp_state_info.keys())

    save_cp_ds_path = os.path.join(new_cp_dir, state_name)
    with open(save_cp_ds_path, 'wb') as fw:
        torch.save(cp_state_info, fw)


def convert_optim_state(pp_rank, total_fragments):
    print(len(total_fragments))

    state_name = f'zero_pp_rank_{pp_rank}_mp_rank_00_optim_states.pt'

    ds_optim_path = os.path.join(ds_ckpt, state_name)
    ds_state_info = torch.load(ds_optim_path, map_location='cpu')
    print('loaded diffusers deepspeed checkpoint')

    cp_optim_path = os.path.join(cp_ckpt, state_name)
    cp_state_info = torch.load(cp_optim_path, map_location='cpu')
    print('loaded composer deepspeed checkpoint')

    for key, cp_state_dict in cp_state_info.items():
        print(f'replace {key}')

        if key == 'optimizer_state_dict':

            for inner_k in cp_state_dict.keys():

                if inner_k == 'loss_scaler':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'dynamic_loss_scale':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'overflow':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'clip_grad':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'base_optimizer_state':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'single_partition_of_fp32_groups':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'zero_stage':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'group_paddings':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'partition_count':
                    cp_state_dict[inner_k] = ds_state_info[key][inner_k]

                elif inner_k == 'ds_version':
                    pass

                elif inner_k == 'param_slice_mappings':

                    for k in cp_state_dict[inner_k][0].keys():
                        df_k = cp2df(k)
                        if df_k is not None:
                            # cp_state_dict[inner_k][0][k] = ds_state_info[key][inner_k][0][df_k]
                            cp_state_dict[inner_k][0][k] = total_fragments[df_k]
                        else:
                            if not k.startswith(
                                    'text_encoder') and not k.startswith('vae'):
                                print(k)

        elif key == 'ds_config':
            cp_state_info[key] = ds_state_info[key]

        elif key == 'ds_version':
            pass

        else:
            print(f'unconvert: {key}')
            __import__('ipdb').set_trace()
            print(f'unconvert: {cp_state_dict}')

    print(cp_state_info.keys())

    save_cp_ds_path = os.path.join(new_cp_dir, state_name)
    with open(save_cp_ds_path, 'wb') as fw:
        torch.save(cp_state_info, fw)

    return cp_state_info


def collect_fragments(world_size):
    total_fragments = OrderedDict()

    ds_state_info_list = []
    for pp_rank in range(world_size):

        state_name = f'zero_pp_rank_{pp_rank}_mp_rank_00_optim_states.pt'

        ds_optim_path = os.path.join(ds_ckpt, state_name)
        ds_state_info = torch.load(ds_optim_path, map_location='cpu')
        print('loaded diffusers deepspeed checkpoint')

        total_fragments.update(ds_state_info['optimizer_state_dict']['param_slice_mappings'][0])
        ds_state_info_list.append(ds_state_info)

    return total_fragments


def convert_all():

    convert_model_states()

    world_size = 2

    # read total fragment address
    total_fragments = collect_fragments(world_size)

    for r in range(world_size):
        convert_optim_state(r, total_fragments)


def make_tarball():
    ...

if __name__ == "__main__":
    # convert_model_states()
    # convert_all()
    make_tarball()

