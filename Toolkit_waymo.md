### Toolkit-Waymo

我们基于mmdetection3d提供了noisy data的toolkit

Waymo 数据集的准备可以参照https://github.com/open-mmlab/mmdetection3d/blob/master/docs/zh_cn/datasets/waymo_det.md

### 生成noise pkl:

Note：也可以直接使用我们提供的pkl

```python
python tools/create_noise_data_waymo.py waymo --root-path  data/waymo --out-dir data/waymo --workers 128 --extra-tag waymo
```

##### 生成的pkl格式：

```python
dict(
	'lidar': dict(...) # 保存lidar的noise信息， 以id检索，例如 1000000
  'camera': dict(...) # 保存camera的noise信息， 以id_cameraid检索， 例如 1000000_1
)
```

对于‘lidar’部分，格式为

```python
dict(
	xxxx: dict(...),
  xxxx: dict(...)
)
# 上述的dict的内容有:
dict(
    # 基本信息
	'prev': 'training/velodyne/xxxx.bin' or '' # 表示为上一帧的lidar文件名字，无上一帧则为空''
    # noise信息
  'noise': dict(
    	'drop_frames': dict(   # 丢帧信息，key为比例，如10,20,...,90
        	'10': dict('discrete': dict('stuck': True or False #是否卡帧, 'replace': 'training/velodyne/xxxx.bin' #替代点云(可为空)
                    'consecutive': dict('stuck': True or False, 'replace': 'training/velodyne/xxxx.bin' # 同上
                  )
            '20': # 同上
            ...
            '90': # 同上
        ),
      'object_failure': True/False     
    )
)
```



对于camera部分，与lidar部分类似，格式为

```python
dict(
	'xxxx_y': dict(...),
  'xxxx_y': dict(...)
)
# 上述的dict的内容有:
dict(
    # 基本信息
    'type': '0' or '1' or ... # 表示哪个方向的相机 0-5
		'prev': 'training/image_y/xxxx.png' or '' # 表示为上一帧的camera文件名字，无上一帧则为空''
    'lidar': dict('file_name': 'training/velodyne/xxxx.bin') # 表示为camera对应的lidar名字
    # noise信息
    'noise': dict(
    	'drop_frames': dict(   # 丢帧信息，key为比例，如10,20,...,90
        	'10': dict('discrete': dict('stuck': True or False #是否卡帧, 'replace': 'training/image_y/xxxx.png' #替代图片(可为空)
                     'consecutive': dict('stuck': True or False, 'replace': 'training/image_y/xxxx.png' # 同上
                  )
            '20': # 同上
            ...
            '90': # 同上
        )
       'extrinsics_noise': dict(
            			'Tr_velo_to_cam':xxx, # 表示noise之前的矩阵
            			'all_Tr_velo_to_cam_noise':xxx, # 表示所有相机一起扰动后的矩阵
            			'single_Tr_velo_to_cam_noise':xxx, # 表示相机单独扰动后的矩阵
             )
       'mask_noise': dict(
            'mask_id': xxx, # 表示第几张mask图片
        )
    )
)
```



### WaymoNoiseDataset

使用方式：将datasets/waymo_dataset.py中WaymoDataset的初始化函数和get_data_info替换为下面的代码，之后在config里面新增一些参数即可

```python
@DATASETS.register_module()
class WaymoNoiseDataset(KittiDataset):
    CLASSES = ('Car', 'Cyclist', 'Pedestrian')
    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 # Add
                 noise_waymo_ann_file = '',
                 extrinsics_noise=False,
                 extrinsics_noise_type='single',
                 drop_frames=False,
                 drop_set=[0, 'discrete'],
                 noise_sensor_type='camera',
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs)

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

        # ADD
        self.extrinsics_noise = extrinsics_noise  # 外参是否扰动
        assert extrinsics_noise_type in ['all', 'single']
        self.extrinsics_noise_type = extrinsics_noise_type  # 外参扰动类型
        self.drop_frames = drop_frames  # 是否丢帧
        self.drop_ratio = drop_set[0]  # 丢帧比例：assert ratio in [10, 20, ..., 90]
        self.drop_type = drop_set[1]  # 丢帧情况：连续(consecutive) or 离散(discrete)
        self.noise_sensor_type = noise_sensor_type  # lidar or camera 丢帧
        noise_data = mmcv.load(noise_waymo_ann_file, file_format='pkl')
        self.noise_camera_data = noise_data['camera']
        if self.extrinsics_noise or self.drop_frames:
            self.noise_data = noise_data[noise_sensor_type]
        else:
            self.noise_data = None
        print('noise setting:')
        if self.drop_frames:
            print('frame drop setting: drop ratio:', self.drop_ratio, ', sensor type:', self.noise_sensor_type,
                  ', drop type:', self.drop_type)
        if self.extrinsics_noise:
            assert noise_sensor_type == 'camera'  ## 只需要相机变化
            print(f'add {extrinsics_noise_type} noise to extrinsics')


    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)

        # ADD
        if self.noise_sensor_type == 'lidar' and self.drop_frames and self.noise_data[sample_idx]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['stuck']:
            replace_file = self.noise_data[sample_idx]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['replace']
            if replace_file != '':
                pts_filename = os.path.join(self.data_root,replace_file)

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []

            for idx_img in range(self.num_views):
                noise_index = str(sample_idx) + '_' + str(idx_img)
                rect = info['calib']['R0_rect'].astype(np.float32)
                if self.extrinsics_noise:
                    Trv2c = self.noise_data[noise_index]['noise']['extrinsics_noise'][f'{self.extrinsics_noise_type}_Tr_velo_to_cam_noise']
                else:
                    Trv2c = self.noise_camera_data[noise_index]['noise']['extrinsics_noise']['Tr_velo_to_cam']

                P0 = info['calib'][f'P{idx_img}'].astype(np.float32)
                lidar2img = P0 @ rect @ Trv2c
                image_path = img_filename.replace('image_0', f'image_{idx_img}')

                if self.noise_sensor_type == 'camera' and self.drop_frames and self.noise_data[noise_index]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['stuck']:
                    replace_file = self.noise_data[noise_index]['noise']['drop_frames'][self.drop_ratio][self.drop_type]['replace']
                    if replace_file:
                      	image_path = os.path.join(self.data_root,replace_file)
                image_paths.append(image_path)
                lidar2img_rts.append(lidar2img)

        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img)

        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts

        # if not self.test_mode:
            # annos = self.get_ann_info(index)
            # input_dict['ann_info'] = annos
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        return input_dict
    
```



### Limited LiDAR FOV

直接将mmdet3d/datasets/pipelines/loading.py的LoadPointsFromFile替换为下面的代码，在config里面对应LoadPointsFromFile的地方传入新增参数point_cloud_angle_range=[-90, 90] or [-60,60]，分别表示只保留前视180和120度

```python
@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk'),
                 #  ADD
                 point_cloud_angle_range=None):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        # ADD
        if point_cloud_angle_range is not None:
            self.filter_by_angle = True
            self.point_cloud_angle_range = point_cloud_angle_range
            print(point_cloud_angle_range)
        else:
            self.filter_by_angle = False

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    # ADD
    def filter_point_by_angle(self, points):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError

        nus_x = points_numpy[:, 1]*-1
        nus_y = points_numpy[:, 0]
        pts_phi = (np.arctan(nus_x/ nus_y) + (
                    nus_y < 0) * np.pi + np.pi * 2) % (np.pi * 2)

        pts_phi[pts_phi > np.pi] -= np.pi * 2
        pts_phi = pts_phi / np.pi * 180

        assert np.all(-180 < pts_phi) and np.all(pts_phi <= 180)

        filt = np.logical_and(pts_phi >= self.point_cloud_angle_range[0], pts_phi <= self.point_cloud_angle_range[1])

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 7.5))
        # ax = Axes3D(fig)
        # ax.view_init(30, 150)
        # ax.scatter(xs=[-54, 54], ys=[-54, 54], zs=[-10, 10], c='white')
        # ax.scatter(xs=points_numpy[:, 0], ys=points_numpy[:, 1], zs=points_numpy[:, 2], c='blue', s=10)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.gca().set_box_aspect((54, 54, 10))
        # plt.savefig('before.png')

        # fig = plt.figure(figsize=(10, 7.5))
        # points_numpy = points_numpy[filt]
        # ax = Axes3D(fig)
        # ax.view_init(30, 150)
        # ax.scatter(xs=[-54, 54], ys=[-54, 54], zs=[-10, 10], c='white')
        # ax.scatter(xs=points_numpy[:, 0], ys=points_numpy[:, 1], zs=points_numpy[:, 2], c='blue', s=2)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.gca().set_box_aspect((54, 54, 10))
        # plt.savefig('after.png')

        return points[filt]

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        # ADD
        if self.filter_by_angle:
            points = self.filter_point_by_angle(points)

        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


```



### Missing of Camera Inputs

直接将mmdet3d/datasets/pipelines/loading.py的LoadMultiViewImageFromFiles替换为下面的代码，在config里面对应LoadMultiViewImageFromFiles的地方传入新增参数drop_camera=[]，list里面就是需要drop的camera

```python
@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged',
                 # ADD
                 drop_camera=[]):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        # ADD
        self.drop_camera = drop_camera

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img

    def __call__(self, results):
        filename = results['img_filename']

        # ADD
        img_lists = []
        for name in filename:
            single_img = mmcv.imread(name, self.color_type)
            if self.img_scale is not None:
                single_img = self.pad(single_img)
            if int(name.split('/')[-2].split('_')[-1]) in self.drop_camera:
                img_lists.append(np.zeros_like(single_img))
            else:
                img_lists.append(single_img)
        img = np.stack(img_lists, axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)



        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = img_lists
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)

```



### Occlusion of Camera Lens 

下载mask图片

```python
@PIPELINES.register_module()
class LoadMaskMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged',
                 # ADD
                 noise_waymo_ann_file='', mask_file=''):
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        
        # ADD
        noise_data = mmcv.load(noise_waymo_ann_file, file_format='pkl')
        self.noise_camera_data = noise_data['camera']
        self.mask_file = mask_file

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            img = np.concatenate([img, np.zeros_like(img[0:1280-886,:])], axis=0)
        return img
    
    # ADD
    def put_mask_on_img(self, img, mask):
        h, w = img.shape[:2]
        mask = np.rot90(mask)
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask / 255
        alpha = np.power(alpha, 3)
        img_with_mask = alpha * img + (1 - alpha) * mask

        return img_with_mask

    def __call__(self, results):
        filename = results['img_filename']

        img_lists = []
        for name in filename:
            single_img = mmcv.imread(name, self.color_type)
            if self.img_scale is not None:
                single_img = self.pad(single_img)
            # ADD
            noise_index = name.split('/')[-1].split('.')[0] + '_' + name.split('/')[-2].split('_')[-1]
            mask_id_png = 'mask_'+ str(self.noise_camera_data[noise_index]['noise']['mask_noise']['mask_id']) + '.jpg'
            mask_name = os.path.join(self.mask_file, mask_id_png)
            mask = mmcv.imread(mask_name, self.color_type)
            single_img = self.put_mask_on_img(single_img, mask)
            img_lists.append(single_img)
        img = np.stack(img_lists, axis=-1)
        
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = [1.0, 1.0]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)
```



### LiDAR Object Failure

在datasets/pipelines/transforms_3d.py中，增加函数Randomdropforeground(记得在datasets/pipelines/init.py中增加对应的引用)

之后在config的test_pipeline的transforms里面，增加Randomdropforeground即可，注意得同时添加dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True)

```python
@PIPELINES.register_module()
class Randomdropforeground(object):
    def __init__(self, noise_waymo_ann_file=''):
        noise_data = mmcv.load(noise_waymo_ann_file, file_format='pkl')
        self.noise_lidar_data = noise_data['lidar']

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        pts_filename = input_dict['pts_filename']
        noise_index = int(pts_filename.split('/')[-1].split('.')[0])

        points = input_dict['points']
        if self.noise_lidar_data[noise_index]['noise']['object_failure']:
            points = self.remove_points_in_boxes(points, gt_bboxes_3d.tensor.numpy())
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += ' fore_drop_rate={})'.format(self.drop_rate)
        return repr_str
```

