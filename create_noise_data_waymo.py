import argparse
from os import path as osp
import random
from math import cos, sin, acos
import math
from pathlib import Path
import mmcv
import numpy as np

def waymo_data_prep(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    convert_kitti=False):
    """Prepare the noisy info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    # Waymo => KITTI
    set_seed(0)
    if convert_kitti:
        from tools.data_converter import waymo_converter as waymo
        splits = ['training', 'validation', 'testing']
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                save_dir = osp.join(out_dir, 'kitti_format', 'training')
            else:
                save_dir = osp.join(out_dir, 'kitti_format', split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split == 'testing'))
            converter.convert()

    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    create_noise_waymo_info_file(out_dir, info_prefix,  workers=workers)

def create_noise_waymo_info_file(data_path,
                           pkl_prefix='waymo',
                           save_path=None,
                           relative_path=True,
                           workers=8):
    """Create noisy info file of waymo dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'waymo'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    waymo_infos_gatherer_noise_val = WaymoNoiseInfoGatherer(
        data_path,
        training=True,
        relative_path=relative_path,
        num_worker=workers)

    frames_drop_ratio_list = [i * 10 for i in range(1, 10)]
    camera_extrinsics_noise = dict(r=(1, 5), t=(0.5 * 0.01, 1.0 * 0.01))
    waymo_infos_noise_val = waymo_infos_gatherer_noise_val.gather(val_img_ids, frames_drop_ratio_list, camera_extrinsics_noise)
    filename = save_path / f'{pkl_prefix}_infos_niose_val.pkl'
    print(f'Waymo noisy info val file is saved to {filename}')
    mmcv.dump(waymo_infos_noise_val, filename)


class WaymoNoiseInfoGatherer:
    def __init__(self,
                 path,
                 extend_matrix=True,
                 training=True,
                 num_worker=8,
                 relative_path=True) -> None:
        self.path = path
        self.training = training
        self.num_worker = num_worker
        self.relative_path = relative_path
        self.extend_matrix = extend_matrix
        self.num_views = 5

    def gather(self, image_ids, frames_drop_ratio_list, camera_extrinsics_noise):
        val_infos_lidar={}
        val_infos_camera={}
        print('filling basic information ......')
        self.base_gather(image_ids, val_infos_lidar, val_infos_camera)

        # LiDAR-stuck/ Camera-stuck
        print('frames drop ratio list', frames_drop_ratio_list)
        for ratio in frames_drop_ratio_list:
            set_seed(0)
            self.drop_frame_gather(image_ids, ratio, val_infos_lidar, val_infos_camera)

        # Spatial Misalignment
        set_seed(0)
        print('camera extrinsics noise range: rotation:', camera_extrinsics_noise['r'], 'translation:',
              camera_extrinsics_noise['t'])
        self.camera_extrinsics_gather(image_ids, camera_extrinsics_noise['r'], camera_extrinsics_noise['t'], val_infos_lidar, val_infos_camera)

        # Occlusion of Camera Lens
        set_seed(0)
        print('allocate mask ...')
        num_mask_type = 16
        self.camera_mask_gather(image_ids, num_mask_type, val_infos_camera)

        # LiDAR Object Failure
        set_seed(0)
        print('allocate object failure ...')
        drop_rate = 0.5
        self.object_failure_gather(image_ids, val_infos_lidar, drop_rate)

        info = dict(lidar=val_infos_lidar, camera=val_infos_camera)
        return info

    def base_gather(self, image_ids, val_infos_lidar, val_infos_camera):
        # mmcv.track_parallel_progress(self.gather_base_single, image_ids, self.num_worker)
        for i, image_id in enumerate(mmcv.track_iter_progress(image_ids)):
            # get lidar infos
            lidar_file_name = get_velodyne_path(
                image_id,
                self.path,
                self.training,
                self.relative_path,
                use_prefix_id=True)
            prev_lidar_file_name = get_velodyne_path(
                image_id - 1,
                self.path,
                self.training,
                self.relative_path,
                exist_check=False,
                use_prefix_id=True)
            if_prev_exists = osp.exists(Path(self.path) / prev_lidar_file_name)
            if not if_prev_exists:
                prev_lidar_file_name = ''
            val_infos_lidar[image_id] = {
                'prev': prev_lidar_file_name,
                'noise': {'drop_frames': dict(),
                          # 'object_failure': True/False
                          },
            }
            # get images infos
            for carema_id in range(5):
                prev_image_path = get_image_path(
                    image_id - 1,
                    self.path,
                    self.training,
                    self.relative_path,
                    exist_check=False,
                    info_type='image_' + str(carema_id),
                    use_prefix_id=True)
                if_prev_exists = osp.exists(
                    Path(self.path) / prev_image_path)
                if not if_prev_exists:
                    prev_image_path = ''

                val_infos_camera[str(image_id) + '_' + str(carema_id)] = {
                    'type': carema_id,
                    'prev': prev_image_path,
                    'lidar': {'file_name': lidar_file_name},
                    'noise': {'drop_frames': dict(),
                              'extrinsics_noise': dict(),
                              'mask_noise': dict()
                              },
                }

    def camera_mask_gather(self, image_ids, num_mask_type, val_infos_camera):
        for image_id in mmcv.track_iter_progress(image_ids):
            for carema_id in range(5):
                mask_id = random.randint(1, num_mask_type)
                mask_noise = {'mask_id': mask_id}
                val_infos_camera[str(image_id) + '_' + str(carema_id)]['noise']['mask_noise'].update(mask_noise)

    def object_failure_gather(self, image_ids, val_infos_lidar, drop_rate):
        for image_id in mmcv.track_iter_progress(image_ids):
            drop_foreground = False
            if np.random.rand() < drop_rate:
                drop_foreground = True
            object_failure = {'object_failure': drop_foreground}
            val_infos_lidar[image_id]['noise'].update(object_failure)

    def drop_frame_gather(self, image_ids, ratio, val_infos_lidar, val_infos_camera):
        num_sample = len(image_ids)
        print('ratio:', ratio, '%')
        set_seed(0)
        discrete_stuck_sample_indicator = {
            'LIDAR': get_discrete_stuck_sample(ratio, num_sample),
            '0': get_discrete_stuck_sample(ratio, num_sample),
            '1': get_discrete_stuck_sample(ratio, num_sample),
            '2': get_discrete_stuck_sample(ratio, num_sample),
            '3': get_discrete_stuck_sample(ratio, num_sample),
            '4': get_discrete_stuck_sample(ratio, num_sample),
        }
        set_seed(0)
        consecutive_stuck_sample_indicator = {
            'LIDAR': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10),
            '0': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10),
            '1': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10),
            '2': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10),
            '3': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10),
            '4': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10),
        }
        # mmcv.track_parallel_progress(self.gather_drop_single, image_ids, self.num_worker):
        for idx, image_id in enumerate(mmcv.track_iter_progress(image_ids)):

            discrete_info = {
                'stuck': discrete_stuck_sample_indicator['LIDAR'][idx],
                'replace': val_infos_lidar[image_id]['prev']
            }

            replace_file = val_infos_lidar[image_id]['prev']
            replace_file_id = image_id - 1
            while (replace_file != '') and \
                    val_infos_lidar[replace_file_id]['noise']['drop_frames'][ratio]['consecutive']['stuck']:
                if val_infos_lidar[replace_file_id]['prev'] == '':
                    break
                replace_file = val_infos_lidar[replace_file_id]['prev']
                replace_file_id = replace_file_id - 1

            consecutive_info = {
                'stuck': consecutive_stuck_sample_indicator['LIDAR'][idx],
                'replace': replace_file
            }

            drop_frame_noise = {
                ratio: {
                    'discrete': discrete_info,
                    'consecutive': consecutive_info,
                }
            }

            val_infos_lidar[image_id]['noise']['drop_frames'].update(drop_frame_noise)

            for carema_id in range(5):
                discrete_info = {
                    'stuck': discrete_stuck_sample_indicator[str(carema_id)][idx],
                    'replace': val_infos_camera[str(image_id) + '_' + str(carema_id)]['prev']
                }

                replace_file = val_infos_camera[str(image_id) + '_' + str(carema_id)]['prev']
                replace_file_id = str(image_id-1) + '_' + str(carema_id)
                while (replace_file != '') and \
                        val_infos_camera[replace_file_id]['noise']['drop_frames'][ratio]['consecutive']['stuck']:
                    if val_infos_camera[replace_file_id]['prev'] == '':
                        break
                    replace_file = val_infos_camera[replace_file_id]['prev']
                    replace_file_id = str(int(replace_file_id.split('_')[0]) - 1) + '_' + str(carema_id)

                consecutive_info = {
                    'stuck': consecutive_stuck_sample_indicator[str(carema_id)][idx],
                    'replace': replace_file
                }

                drop_frame_noise = {
                    ratio: {
                        'discrete': discrete_info,
                        'consecutive': consecutive_info,
                    }
                }
                val_infos_camera[str(image_id) + '_' + str(carema_id)]['noise']['drop_frames'].update(
                    drop_frame_noise)

    # r=(1, 5), t=(0.5*0.01, 1.0*0.01)
    def camera_extrinsics_gather(self, image_ids, rot_noise_range, trans_noise_range, val_infos_lidar, val_infos_camera):
        self.rot_noise_range = rot_noise_range
        self.trans_noise_range = trans_noise_range
        # image_infos = mmcv.track_parallel_progress(self.gather_noise_single, image_ids, self.num_worker)
        for idx, image_id in enumerate(mmcv.track_iter_progress(image_ids)):
            calib_info = {}
            calib_path = get_calib_path(
                image_id,
                self.path,
                self.training,
                relative_path=False,
                use_prefix_id=True)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            all_noise_r = np.array(get_noise_rot_mat(self.rot_noise_range))
            all_noise_t = np.array(get_noise_trans(self.trans_noise_range))
            for carema_id, line_num in enumerate(range(6, 6 + self.num_views)):
                trans = np.array([float(info) for info in lines[line_num].split(' ')[1:13]]).reshape(3, 4)
                single_noise_trans, all_noise_trans = self.get_noise_extrinsics_single_and_all(trans, all_noise_r, all_noise_t)
                if self.extend_matrix:
                    trans = _extend_matrix(trans)
                calib_info['Tr_velo_to_cam'] = trans
                calib_info['single_Tr_velo_to_cam_noise'] = single_noise_trans
                calib_info['all_Tr_velo_to_cam_noise'] = all_noise_trans

                val_infos_camera[str(image_id) + '_' + str(carema_id)]['noise']['extrinsics_noise'].update(calib_info)

    def get_noise_extrinsics(self, Tr_velo_to_cam, noise_r, noise_t):
        # Tr_velo_to_cam = （R | T）
        T_velo_to_cam = Tr_velo_to_cam[:, -1]
        R_velo_to_cam = Tr_velo_to_cam[:, :-1]
        R_velo_to_cam_with_noise = R_velo_to_cam @ noise_r
        T_velo_to_cam_with_noise = T_velo_to_cam + noise_t
        T_velo_to_cam_with_noise = T_velo_to_cam_with_noise[:, None]
        Tr_velo_to_cam_with_noise = np.concatenate((R_velo_to_cam_with_noise, T_velo_to_cam_with_noise), axis=1)
        if self.extend_matrix:
            Tr_velo_to_cam_with_noise = _extend_matrix(Tr_velo_to_cam_with_noise)
        return Tr_velo_to_cam_with_noise

    def get_noise_extrinsics_single_and_all(self, Tr_velo_to_cam, all_noise_r, all_noise_t):
        noise_r = np.array(get_noise_rot_mat(self.rot_noise_range))
        noise_t = np.array(get_noise_trans(self.trans_noise_range))
        return self.get_noise_extrinsics(Tr_velo_to_cam, noise_r, noise_t), \
               self.get_noise_extrinsics(Tr_velo_to_cam, all_noise_r, all_noise_t)


def get_discrete_stuck_sample(ratio, num_sample):
    id_list = [i for i in range(num_sample)]
    stuck_list = [False] * num_sample

    num_stuck = num_sample * ratio // 100
    random.shuffle(id_list)
    id_list = id_list[:num_stuck]
    for i in id_list:
        stuck_list[i] = True
    return stuck_list

# waymo 10Hz
def get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=10):
    # ratio = ratio/consecutive_len
    ratio = 1 - math.pow(1 - ratio / 100., 1.0 / consecutive_len)
    id_list = [i for i in range(num_sample + consecutive_len)]
    stuck_list = [False] * (num_sample + consecutive_len)
    # num_stuck = int(num_sample * ratio / 100)
    num_stuck = int(num_sample * ratio)
    random.shuffle(id_list)
    id_list = id_list[:num_stuck]
    for i in id_list:
        for k in range(consecutive_len):
            if i + k < num_sample + consecutive_len:
                stuck_list[i + k] = True
    return stuck_list[consecutive_len:]

def get_random_axis():
    u_x = random.random()
    u_y = random.random()
    theta = acos(1 - 2 * u_x)
    phi = 2 * math.pi * u_y

    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    return [x, y, z]

def get_noise_rot_mat(noise_range):
    rot_axis = get_random_axis()

    x, y, z = rot_axis
    a, b = noise_range[0], noise_range[1]
    noise_theta = a + (b - a) * random.random()
    noise_theta = noise_theta / 360 * math.pi

    if random.choices([True, False])[0]:
        noise_theta *= -1

    rot_mat = [
        [x * x * (1 - cos(noise_theta)) + cos(noise_theta), x * y * (1 - cos(noise_theta)) + z * sin(noise_theta),
         x * z * (1 - cos(noise_theta)) - y * sin(noise_theta)],
        [x * y * (1 - cos(noise_theta)) - z * sin(noise_theta), y * y * (1 - cos(noise_theta)) + cos(noise_theta),
         y * z * (1 - cos(noise_theta)) + x * sin(noise_theta)],
        [x * z * (1 - cos(noise_theta)) + y * sin(noise_theta), y * z * (1 - cos(noise_theta)) - x * sin(noise_theta),
         z * z * (1 - cos(noise_theta)) + cos(noise_theta)]
    ]

    return rot_mat

def get_noise_trans(noise_range):
    a, b = noise_range[0], noise_range[1]
    x = a + (b - a) * random.random()
    if random.choices([True, False])[0]:
        x *= -1

    y = a + (b - a) * random.random()
    if random.choices([True, False])[0]:
        y *= -1

    z = a + (b - a) * random.random()
    if random.choices([True, False])[0]:
        z *= -1

    return [x, y, z]

def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)

def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True,
                        exist_check=True,
                        use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def get_image_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2',
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, info_type, '.png', training,
                               relative_path, exist_check, use_prefix_id)

def get_calib_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)

def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers,
            convert_kitti=False
        )


'''
python tools/create_noise_data_waymo.py waymo --root-path  data/waymo --out-dir data/waymo --workers 128 --extra-tag waymo

'''