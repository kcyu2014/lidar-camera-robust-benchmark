import argparse
from fileinput import filename
import os
from os import path as osp
import numpy as np
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database
import random
import pickle
import mmcv
import math
from math import cos, sin, acos
from pyquaternion import Quaternion


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10,
                       mmdet_convert=False):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """

    if mmdet_convert:
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps)

        if version == 'v1.0-test':
            return

        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)
        create_groundtruth_database(dataset_name, root_path, info_prefix,
                                    f'{out_dir}/{info_prefix}_infos_train.pkl')

    create_noise_nuscenes_info_file(root_path, version=version)

def create_noise_nuscenes_info_file(root_path, version):


    print('Generate noise info. this may take several minutes.')
    
    nuscenes_infos_gatherer_noise_val = NuScenesNoiseInfoGatherer(
                                                        root_path, 
                                                        version=version)

    set_seed(0)
    frames_drop_ratio_list = [i*10 for i in range(1, 10)]
    camera_extrinsics_noise = dict(r=(1, 5), t=(0.5*0.01, 1.0*0.01))

    noise_info = nuscenes_infos_gatherer_noise_val.gather(frames_drop_ratio_list, camera_extrinsics_noise)
    
    print('Generate mmdet info.')
    mmdet_info_path = osp.join(root_path, f'nuscenes_infos_val.pkl')
    lidar_noise_info = noise_info['lidar']

    mmdet_file = open(mmdet_info_path, 'rb')
    mmdet_info = pickle.load(mmdet_file)['infos']

    for it in mmdet_info:
        lidar_name = it['lidar_path'].split('/')[-1]
        lidar_noise_info[lidar_name]['mmdet_info'] = it
    noise_info['lidar'] = lidar_noise_info
    filename = osp.join(root_path, f'nuscenes_infos_val_with_noise.pkl')

    print(f'nuScenes noisy info val file is saved to {filename}')
    mmcv.dump(noise_info, filename)


class NuScenesNoiseInfoGatherer:
    def __init__(self,
                 root_path,
                 version='v1.0-trainval') -> None:
        
        self.root_path = root_path
        self.version = version
    
    def gather(self, frames_drop_ratio_list, camera_extrinsics_noise):
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.root_path, verbose=True)
        from nuscenes.utils import splits
        available_vers = ['v1.0-trainval']
        assert self.version in available_vers
        if self.version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        else:
            raise ValueError('unknown')

        # filter existing scenes.
        available_scenes = self.get_available_scenes()
        available_scene_names = [s['name'] for s in available_scenes]
        self.val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        
        self.val_scenes = set([
            available_scenes[available_scene_names.index(s)]['token']
            for s in self.val_scenes
        ])

        print('val scene: {}'.format(len(self.val_scenes)))

        val_nusc_infos_camera = dict()
        val_nusc_infos_lidar = dict()

        print('filling basic information ......')
        num_sample = self.base_gather(val_nusc_infos_lidar, val_nusc_infos_camera)

        # random drop frames
        print('frames drop ratio list', frames_drop_ratio_list)
        for ratio in frames_drop_ratio_list:
            print('ratio:', ratio, '%')
            set_seed(0)
            self.drop_frame_gather(ratio, val_nusc_infos_lidar, val_nusc_infos_camera, num_sample)
            
        
        # random add noise for camera extrinsics
        set_seed(0)
        print('camera extrinsics noise range: rotation:', camera_extrinsics_noise['r'], 'translation:', camera_extrinsics_noise['t'])
        
        self.camera_extrinsics_gather(camera_extrinsics_noise, val_nusc_infos_camera)

        # random mask
        set_seed(0)
        print('allocate mask ...')
        num_mask_type = 16
        self.camera_mask_gather(num_mask_type, val_nusc_infos_camera)

        set_seed(0)
        print('allocate object failure ...')
        drop_rate = 0.5
        self.object_failure_gather(val_nusc_infos_lidar, drop_rate)

        # collect noise information
        data = dict(lidar=val_nusc_infos_lidar, camera=val_nusc_infos_camera, version=self.version)

        # mmcv.dump(data, output_path)
        return data
    
    def get_available_scenes(self):
        """Get available scenes from the input nuscenes class.

        Given the raw data, get the information of available scenes for
        further info generation.

        Args:
            nusc (class): Dataset class in the nuScenes dataset.

        Returns:
            available_scenes (list[dict]): List of basic information for the
                available scenes.
        """
        available_scenes = []
        print('total scene num: {}'.format(len(self.nusc.scene)))
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = self.nusc.get_sample_data(sd_rec['token'])
                lidar_path = str(lidar_path)
                if os.getcwd() in lidar_path:
                    # path from lyftdataset is absolute path
                    lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                    # relative path
                if not mmcv.is_filepath(lidar_path):
                    scene_not_exist = True
                    break
                else:
                    break
            if scene_not_exist:
                continue
            available_scenes.append(scene)
        print('exist scene num: {}'.format(len(available_scenes)))
        return available_scenes

    def base_gather(self, val_nusc_infos_lidar, val_nusc_infos_camera):
        camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
        
        num_sample = 0
        
        for sample in mmcv.track_iter_progress(self.nusc.sample):

            if sample['scene_token'] not in self.val_scenes:
                continue
            num_sample += 1
            
            # get lidar information
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path, _, _ = self.nusc.get_sample_data(lidar_token)
            lidar_file_name = lidar_path.split('/')[-1]

            prev_token = sample['prev']
            if not prev_token == '': 
                prev_sample = self.nusc.get('sample', prev_token)
                prev_lidar_token = prev_sample['data']['LIDAR_TOP']
                prev_lidar_path, _, _ = self.nusc.get_sample_data(prev_lidar_token)
                prev_lidar_file_name = prev_lidar_path.split('/')[-1]
            else:
                prev_lidar_file_name = ''

            cam_info_for_lidar = dict()
            for cam in camera_types:
                cam_token = sample['data'][cam]
                sd_rec = self.nusc.get('sample_data', cam_token)
                cam_path = str(self.nusc.get_sample_data_path(sd_rec['token']))
                cam_file_name = cam_path.split('/')[-1]
                cam_info_for_lidar[cam] = {'file_name':cam_file_name}

            val_nusc_infos_lidar[lidar_file_name] = {
                'prev':prev_lidar_file_name,
                'cam': cam_info_for_lidar,
                'noise': {'drop_frames': dict()},
            }
            
            # get camera infomation
            for cam in camera_types:

                cam_token = sample['data'][cam]
                sd_rec = self.nusc.get('sample_data', cam_token)
                cam_path = str(self.nusc.get_sample_data_path(sd_rec['token']))
                cam_file_name = cam_path.split('/')[-1]
                

                # get prev frames
                prev_token = sample['prev']
                if not prev_token == '': 
                    prev_sample = self.nusc.get('sample', prev_token)
                    prev_cam_token = prev_sample['data'][cam]
                    prev_sd_rec = self.nusc.get('sample_data', prev_cam_token)
                    prev_cam_path = str(self.nusc.get_sample_data_path(prev_sd_rec['token']))
                    prev_cam_file_name = prev_cam_path.split('/')[-1]
                else:
                    prev_cam_file_name = ''

                val_nusc_infos_camera[cam_file_name] = {
                    'type': cam,
                    'prev': prev_cam_file_name,
                    'lidar': {'file_name':lidar_file_name},
                    'noise': {'drop_frames': dict(),
                              'extrinsics_noise': dict(),
                              'mask_noise': dict()
                             },
                    }
        return num_sample

    def object_failure_gather(self, val_nusc_infos_lidar, drop_rate):
        set_seed(0)
        for sample in mmcv.track_iter_progress(self.nusc.sample):

            if sample['scene_token'] not in self.val_scenes:
                continue
                
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path, _, _ = self.nusc.get_sample_data(lidar_token)
            lidar_file_name = lidar_path.split('/')[-1]

            drop_foreground = False
            if np.random.rand() < drop_rate:
                drop_foreground = True
            object_failure = {'object_failure': drop_foreground}

            val_nusc_infos_lidar[lidar_file_name]['noise'].update(object_failure)

    def drop_frame_gather(self,
                          ratio, 
                          val_nusc_infos_lidar, 
                          val_nusc_infos_camera,
                          num_sample):
        camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
        
        set_seed(0)
        discrete_stuck_sample_indicator = {
            'LIDAR': get_discrete_stuck_sample(ratio, num_sample),
            'CAM_FRONT': get_discrete_stuck_sample(ratio, num_sample),
            'CAM_FRONT_RIGHT': get_discrete_stuck_sample(ratio, num_sample),
            'CAM_FRONT_LEFT': get_discrete_stuck_sample(ratio, num_sample),
            'CAM_BACK': get_discrete_stuck_sample(ratio, num_sample),
            'CAM_BACK_LEFT': get_discrete_stuck_sample(ratio, num_sample),
            'CAM_BACK_RIGHT': get_discrete_stuck_sample(ratio, num_sample),
        }
        set_seed(0)
        consecutive_stuck_sample_indicator = {
            'LIDAR': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
            'CAM_FRONT': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
            'CAM_FRONT_RIGHT': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
            'CAM_FRONT_LEFT': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
            'CAM_BACK': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
            'CAM_BACK_LEFT': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
            'CAM_BACK_RIGHT': get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4),
        }
        sample_id = 0
        for sample in mmcv.track_iter_progress(self.nusc.sample):

            if sample['scene_token'] not in self.val_scenes:
                continue
                
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path, _, _ = self.nusc.get_sample_data(lidar_token)
            lidar_file_name = lidar_path.split('/')[-1]

            discrete_info = {
                'stuck': discrete_stuck_sample_indicator['LIDAR'][sample_id],
                'replace': val_nusc_infos_lidar[lidar_file_name]['prev']
            }

            replace_file = val_nusc_infos_lidar[lidar_file_name]['prev']

            while (replace_file != '') and val_nusc_infos_lidar[replace_file]['noise']['drop_frames'][ratio]['consecutive']['stuck']:
                if val_nusc_infos_lidar[replace_file]['prev'] == '':
                        break
                replace_file = val_nusc_infos_lidar[replace_file]['prev']

            consecutive_info = {
                'stuck': consecutive_stuck_sample_indicator['LIDAR'][sample_id],
                'replace': replace_file
            }


            drop_frame_noise = {
                ratio:{
                    'discrete': discrete_info,
                    'consecutive': consecutive_info,
                }
            }
            val_nusc_infos_lidar[lidar_file_name]['noise']['drop_frames'].update(drop_frame_noise)


            # obtain 6 image's information per frame
            for cam in camera_types:

                cam_token = sample['data'][cam]
                sd_rec = self.nusc.get('sample_data', cam_token)
                cam_path = str(self.nusc.get_sample_data_path(sd_rec['token']))
                cam_file_name = cam_path.split('/')[-1]

                discrete_info = {
                    'stuck': discrete_stuck_sample_indicator[cam][sample_id],
                    'replace': val_nusc_infos_camera[cam_file_name]['prev']
                }

                replace_file = val_nusc_infos_camera[cam_file_name]['prev']

                while (replace_file != '') and val_nusc_infos_camera[replace_file]['noise']['drop_frames'][ratio]['consecutive']['stuck']:
                    if val_nusc_infos_camera[replace_file]['prev'] == '':
                        break
                    replace_file = val_nusc_infos_camera[replace_file]['prev']

                consecutive_info = {
                    'stuck': consecutive_stuck_sample_indicator[cam][sample_id],
                    'replace': replace_file
                }

                drop_frame_noise = {
                    ratio:{
                        'discrete': discrete_info,
                        'consecutive': consecutive_info,
                        }
                }
                val_nusc_infos_camera[cam_file_name]['noise']['drop_frames'].update(drop_frame_noise)

                
            sample_id += 1



    def camera_extrinsics_gather(self,
                                 camera_extrinsics_noise,
                                 val_nusc_infos_camera):

        rot_noise_range = camera_extrinsics_noise['r']
        trans_noise_range = camera_extrinsics_noise['t']
        set_seed(0)
        for sample in mmcv.track_iter_progress(self.nusc.sample):

            if sample['scene_token'] not in self.val_scenes:
                continue
            lidar_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = self.nusc.get('calibrated_sensor',
                                lidar_rec['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', lidar_rec['ego_pose_token'])

            l2e_r = cs_record['rotation']
            l2e_t = cs_record['translation']
            e2g_r = pose_record['rotation']
            e2g_t = pose_record['translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix


            # obtain 6 image's information per frame
            camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
            noise_r_all = get_noise_rot_mat(rot_noise_range)
            noise_t_all = get_noise_trans(trans_noise_range)

            noise_r_all = np.array(noise_r_all)
            noise_t_all = np.array(noise_t_all)
            for cam in camera_types:

                cam_token = sample['data'][cam]
                sd_rec = self.nusc.get('sample_data', cam_token)
                cam_path = str(self.nusc.get_sample_data_path(sd_rec['token']))
                cam_file_name = cam_path.split('/')[-1]
                
                noise_r = get_noise_rot_mat(rot_noise_range)
                noise_t = get_noise_trans(trans_noise_range)

                noise_r = np.array(noise_r)
                noise_t = np.array(noise_t)
                
                cam_info = self.obtain_noise_sensor2top(cam_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, noise_r, noise_t, noise_name='single')
                cam_info.update(self.obtain_noise_sensor2top(cam_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, noise_r_all, noise_t_all, noise_name='all'))
                
                cam_info.update(self.obtain_sensor2top(cam_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat))
                
                val_nusc_infos_camera[cam_file_name]['noise']['extrinsics_noise'].update(cam_info)


    def obtain_noise_sensor2top(self,
                                sensor_token,
                                l2e_t,
                                l2e_r_mat,
                                e2g_t,
                                e2g_r_mat,
                                noise_r,
                                noise_t,
                                noise_name):
        """Obtain the info with RT matric from general sensor to Top LiDAR.

        Args:
            nusc (class): Dataset class in the nuScenes dataset.
            sensor_token (str): Sample data token corresponding to the
                specific sensor type.
            l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
            l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
                in shape (3, 3).
            e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
            e2g_r_mat (np.ndarray): Rotation matrix from ego to global
                in shape (3, 3).
            sensor_type (str): Sensor to calibrate. Default: 'lidar'.

        Returns:
            sweep (dict): Sweep information after transformation.
        """
        sd_rec = self.nusc.get('sample_data', sensor_token)
        cs_record = self.nusc.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        s2e_r = cs_record['rotation']
        s2e_t = cs_record['translation']

        s2e_r_mat = Quaternion(s2e_r).rotation_matrix

        # s2e_r_mat_with_noise = s2e_r_mat@noise_r
        s2e_r_mat_with_noise = noise_r@s2e_r_mat
        s2e_r_with_noise = Quaternion(matrix=s2e_r_mat_with_noise)
        s2e_r_with_noise = np.array(s2e_r_with_noise.elements)

        # s2e_t_with_noise = s2e_t + noise_t
        s2e_t_with_noise = noise_t + s2e_t@noise_r.T

        sweep = {
            f'{noise_name}_noise_sensor2ego_translation': s2e_t_with_noise,
            f'{noise_name}_noise_sensor2ego_rotation': s2e_r_with_noise,
            f'{noise_name}_noise_ego2global_translation': pose_record['translation'],
            f'{noise_name}_noise_ego2global_rotation': pose_record['rotation'],
        }
        # print(sweep)
        l2e_r_s = sweep[f'{noise_name}_noise_sensor2ego_rotation']
        l2e_t_s = sweep[f'{noise_name}_noise_sensor2ego_translation']
        e2g_r_s = sweep[f'{noise_name}_noise_ego2global_rotation']
        e2g_t_s = sweep[f'{noise_name}_noise_ego2global_translation']

        # obtain the RT from sensor to Top LiDAR
        # sweep->ego->global->ego'->lidar
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        # print(l2e_r_s_mat,e2g_r_s_mat)
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
        sweep[f'{noise_name}_noise_sensor2lidar_rotation'] = R.T  # points @ R.T + T
        sweep[f'{noise_name}_noise_sensor2lidar_translation'] = T

        return sweep

    def obtain_sensor2top(self,
                          sensor_token,
                          l2e_t,
                          l2e_r_mat,
                          e2g_t,
                          e2g_r_mat):
        """Obtain the info with RT matric from general sensor to Top LiDAR.

        Args:
            nusc (class): Dataset class in the nuScenes dataset.
            sensor_token (str): Sample data token corresponding to the
                specific sensor type.
            l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
            l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
                in shape (3, 3).
            e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
            e2g_r_mat (np.ndarray): Rotation matrix from ego to global
                in shape (3, 3).
            sensor_type (str): Sensor to calibrate. Default: 'lidar'.

        Returns:
            sweep (dict): Sweep information after transformation.
        """
        sd_rec = self.nusc.get('sample_data', sensor_token)
        cs_record = self.nusc.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        s2e_r = cs_record['rotation']
        s2e_t = cs_record['translation']

        sweep = {
            'sensor2ego_translation': s2e_t,
            'sensor2ego_rotation': s2e_r,
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
        }
        # print(sweep)
        l2e_r_s = sweep['sensor2ego_rotation']
        l2e_t_s = sweep['sensor2ego_translation']
        e2g_r_s = sweep['ego2global_rotation']
        e2g_t_s = sweep['ego2global_translation']

        # obtain the RT from sensor to Top LiDAR
        # sweep->ego->global->ego'->lidar
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        # print(l2e_r_s_mat,e2g_r_s_mat)
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
        sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
        sweep['sensor2lidar_translation'] = T

        return sweep


    def camera_mask_gather(self, 
                           num_mask_type,
                           val_nusc_infos_camera):
        camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
        
        set_seed(0)
        
        for sample in mmcv.track_iter_progress(self.nusc.sample):

            if sample['scene_token'] not in self.val_scenes:
                continue
                
            for cam in camera_types:

                cam_token = sample['data'][cam]
                sd_rec = self.nusc.get('sample_data', cam_token)
                cam_path = str(self.nusc.get_sample_data_path(sd_rec['token']))
                cam_file_name = cam_path.split('/')[-1]

                mask_id = random.randint(1, num_mask_type)
                mask_noise = {'mask_id': mask_id}

                val_nusc_infos_camera[cam_file_name]['noise']['mask_noise'].update(mask_noise)



def get_random_axis():
    u_x = random.random()
    u_y = random.random()
    theta = acos(1 - 2*u_x)
    phi = 2 * math.pi * u_y

    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    z = cos(theta)
    return [x, y, z]


def get_noise_rot_mat(noise_range):
    rot_axis = get_random_axis()

    x, y, z = rot_axis
    a, b = noise_range[0], noise_range[1] 
    noise_theta = a + (b - a) * random.random()
    noise_theta = noise_theta/360 * math.pi

    if random.choices([True, False])[0]:
        noise_theta *= -1

    rot_mat = [
        [x*x*(1 - cos(noise_theta)) + cos(noise_theta), x*y*(1 - cos(noise_theta)) + z*sin(noise_theta), x*z*(1 - cos(noise_theta)) - y*sin(noise_theta)],
        [x*y*(1 - cos(noise_theta)) - z*sin(noise_theta), y*y*(1 - cos(noise_theta)) + cos(noise_theta), y*z*(1 - cos(noise_theta)) + x*sin(noise_theta)],
        [x*z*(1 - cos(noise_theta)) + y*sin(noise_theta), y*z*(1 - cos(noise_theta)) - x*sin(noise_theta), z*z*(1 - cos(noise_theta)) + cos(noise_theta)]
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


def get_discrete_stuck_sample(ratio, num_sample):
    id_list = [i for i in range(num_sample)]
    stuck_list = [False] * num_sample

    num_stuck = num_sample*ratio//100
    random.shuffle(id_list)
    id_list = id_list[:num_stuck]
    for i in id_list:
        stuck_list[i] = True
    
    return stuck_list

def get_consecutive_stuck_sample(ratio, num_sample, consecutive_len=4):
    # ratio = ratio/consecutive_len
    ratio = 1 - math.pow(1 - ratio/100., 1.0/consecutive_len)
    id_list = [i for i in range(num_sample+consecutive_len)]
    stuck_list = [False] * (num_sample+consecutive_len)

    num_stuck = int(num_sample*ratio)
    random.shuffle(id_list)
    id_list = id_list[:num_stuck]
    for i in id_list:
        for k in range(consecutive_len):
            if i+k<num_sample+consecutive_len:
                stuck_list[i+k] = True
    
    return stuck_list[consecutive_len:]


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
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    # python tools/create_noise_data_nuscenes.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    
    # elif args.dataset == 'waymo':
    #     waymo_data_prep(
    #         root_path=args.root_path,
    #         info_prefix=args.extra_tag,
    #         version=args.version,
    #         out_dir=args.out_dir,
    #         workers=args.workers,
    #         max_sweeps=args.max_sweeps)

