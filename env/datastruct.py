import numpy as np


#  -------------------------------------TimeSlot-------------------------------------  #
class TimeSlot(object):
    """ 时隙属性及其操作 """

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.slot_length = self.end - self.start

        self.now = start
        self.reset()

    def add_time(self) -> None:
        """ 增加时间 """
        self.now += 1

    def is_end(self) -> bool:
        """ 检查时隙是否结束 """
        return self.now >= self.end

    def get_slot_length(self) -> int:
        """ 返回时隙长度 """
        return self.slot_length

    def get_now(self) -> int:
        """ 返回当前时隙 """
        return self.now

    def reset(self) -> None:
        """ 重置时隙 """
        self.now = self.start


#  --------------------------------需要解决的任务--------------------------------  #
class Function(object):
    """ 任务属性及其操作 """

    def __init__(self, input_size: float, difficulty: float, output_size: float, delay_threshold: int) -> None:
        self.input_size = input_size
        self.difficulty = difficulty
        self.output_size = output_size
        self.delay_threshold = delay_threshold

    def get_input_size(self) -> float:
        return float(self.input_size)

    def get_difficulty(self) -> float:
        return float(self.difficulty)

    def get_output_size(self) -> float:
        return float(self.output_size)

    def get_delay_threshold(self) -> int:
        return int(self.delay_threshold)


#  -------------------------------------Vehicle-------------------------------------  #
class Vehicle(object):
    """ 车辆属性及其操作 """

    def __init__(
            self,
            road_range: int,
            min_init_queue_length: float,
            max_init_queue_length: float,
            min_drive_speed: float,
            max_drive_speed: float,
            min_compute_speed: float,
            max_compute_speed: float
    ) -> None:
        self.compute_speed = np.random.uniform(min_compute_speed, max_compute_speed)  # 计算速度
        self.vehicle_speed = np.random.uniform(min_drive_speed, max_drive_speed)  # 车辆速度
        self.stay_time = int(road_range / self.vehicle_speed)  # 停留时间
        self.task_queue = np.random.uniform(low=min_init_queue_length, high=max_init_queue_length)

    def stay_time_update(self) -> None:
        """ 更新停留时间 """
        self.stay_time -= 1

    def task_in(self, task_size):
        """ 任务入队 """
        self.task_queue += task_size
        return task_size

    def task_execute(self) -> float:
        """ 车辆任务执行 """
        out_datasize = self.compute_speed
        task_completion_amount = min(out_datasize, self.task_queue)
        self.task_queue -= task_completion_amount
        return task_completion_amount


class VehicleList(object):
    """ 车辆队列属性及其操作 """

    def __init__(
            self,
            road_range: int,
            min_init_queue_length: float,
            max_init_queue_length: float,
            min_drive_speed: float,
            max_drive_speed: float,
            min_compute_speed: float,
            max_compute_speed: float,
            vehicle_number: int
    ) -> None:
        self.vehicle_number = vehicle_number

        # 单个车辆生成所需，并且需要在别的函数内调用
        self.road_range = road_range
        self.min_init_queue_length = min_init_queue_length
        self.max_init_queue_length = max_init_queue_length
        self.min_vehicle_speed = min_drive_speed
        self.max_vehicle_speed = max_drive_speed
        self.min_vehicle_compute_speed = min_compute_speed
        self.max_vehicle_compute_speed = max_compute_speed

        # 车辆队列初始化
        self.vehicle_list = [Vehicle(road_range=self.road_range, min_init_queue_length=self.min_init_queue_length,
                                     max_init_queue_length=self.max_init_queue_length,
                                     min_drive_speed=self.min_vehicle_speed, max_drive_speed=self.max_vehicle_speed,
                                     min_compute_speed=self.min_vehicle_compute_speed,
                                     max_compute_speed=self.max_vehicle_compute_speed) for _ in
                             range(self.vehicle_number)]

    def update(self) -> bool:
        # 时间更新
        for vehicle in self.vehicle_list:
            vehicle.stay_time_update()

        # 车辆驶出
        self.vehicle_list = [v for v in self.vehicle_list if v.stay_time > 0]

        # 车辆驶入
        space_left = self.vehicle_number - len(self.vehicle_list)
        self.vehicle_list += [Vehicle(road_range=self.road_range, min_init_queue_length=self.min_init_queue_length,
                                      max_init_queue_length=self.max_init_queue_length,
                                      min_drive_speed=self.min_vehicle_speed, max_drive_speed=self.max_vehicle_speed,
                                      min_compute_speed=self.min_vehicle_compute_speed,
                                      max_compute_speed=self.max_vehicle_compute_speed)
                              for _ in range(space_left)]

        if len(self.vehicle_list) == self.vehicle_number:
            return True
        else:
            return False


#  ---------------------------------------RSU---------------------------------------  #
class RSU(object):
    """ RSU属性及其操作 """

    def __init__(
            self,
            min_init_queue_length: float,
            max_init_queue_length: float,
            compute_speed: float,
    ) -> None:
        # 计算速度和存储容量
        self.compute_speed = compute_speed  # 计算速度
        self.task_queue = np.random.uniform(low=min_init_queue_length, high=max_init_queue_length)

    def task_in(self, task_size):
        """ 任务入队 """
        self.task_queue += task_size
        return task_size

    def task_execute(self) -> float:
        """ RSU任务执行 """
        out_datasize = self.compute_speed
        task_completion_amount = min(out_datasize, self.task_queue)
        self.task_queue -= task_completion_amount
        return task_completion_amount


class RSUList(object):
    """ RSU队列属性及其操作 """

    def __init__(
            self,
            min_init_queue_length: float,
            max_init_queue_length: float,
            compute_speed: float,
            rsu_number
    ):
        self.rsu_number = rsu_number
        self.rsu_list = [RSU(min_init_queue_length,
                             max_init_queue_length,
                             compute_speed) for _ in range(self.rsu_number)]
