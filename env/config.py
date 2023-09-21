import numpy as np


class VehicularEnvConfig:
    """ 道路场景参数设置 """

    def __init__(self):
        # 道路信息
        self.road_range: int = 500  # 500m

        # 时间信息
        self.start_time: int = 0
        self.end_time: int = 199

        # 车辆信息
        self.vehicle_number_dict: dict = {"number_1": 4, "number_2": 6, "number_3": 8, "number_4": 10}
        self.min_vehicle_drive_speed: float = 10.  # 10m/s
        self.max_vehicle_drive_speed: float = 20.  # 10m/s

        self.min_vehicle_compute_speed: float = 3  # 2MB/s
        self.max_vehicle_compute_speed: float = 3  # 3MB/s

        self.min_vehicle_init_queue_length = 5  # 0MB
        self.max_vehicle_init_queue_length = 10  # 0MB

        self.min_vehicle_task_flow: float = 2.8  # 0MB
        self.max_vehicle_task_flow: float = 2.8  # 3MB
        self.vehicle_task_flow: float = np.random.uniform(self.min_vehicle_task_flow, self.max_vehicle_task_flow)

        # RSU信息
        self.rsu_number: int = 3

        self.rsu_compute_speed: dict = {"speed_1": 6, "speed_2": 7, "speed_3": 8, "speed_4": 9}

        self.min_rsu_init_queue_length: float = 20  # 0MB
        self.max_rsu_init_queue_length: float = 40  # 0MB

        self.min_rsu_task_flow: float = 5  # 0MB
        self.max_rsu_task_flow: float = 5  # 7MB
        self.rsu_task_flow: float = np.random.uniform(self.min_rsu_task_flow, self.max_rsu_task_flow)

        # 通讯信息
        self.theta_rv: float = 3.5  # 3.5s/MB
        self.theta_rc: float = 4.8  # 4.8s/MB
        self.theta_vc: float = 8  # 8s/MB

        # 任务信息
        self.min_delay_threshold: float = 4
        self.max_delay_threshold: float = 6
        self.min_in_datasize: float = 1  # 1MB
        self.max_in_datasize: float = 2  # 2MB
        self.min_out_datasize: float = 0.08  # 0.08MB
        self.max_out_datasize: float = 0.18  # 0.18MB

        # 模型信息
        self.w: dict = {"weight_1": 40, "weight_2": 60, "weight_3": 80, "weight_4": 100}
        self.reward_threshold: dict = {"weight_1": -1500, "weight_2": -1500, "weight_3": -1500, "weight_4": -1500}
