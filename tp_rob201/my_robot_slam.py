"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # points to explore
        self.waypoints = [
            [0, -400, 0],      # Ponto 1: frente
            [-200, -500, 0],    # Ponto 2: diagonal direita
            [-500, -500, 0],      # Ponto 3: esquerda
            [-200, -500, 0],     # Ponto 4: trás
            [-200, -200, 0],
            [-400, -100, 0],
            [-200, 50, 0],
            [-600, 0, 0],
            [-800, -100, 0],
            [-600, -200, 0],
            [-800, -200, 0],
            [-850, -350, 0],
            [-800, -300, 0],
            [-900, -250, 0],
            [-900, 0, 0]
                                  
        ]
        self.current_waypoint_idx = 0
        self.distance_threshold = 50  


    
    def control(self):
        """
        Main control function executed at each time step
        """
        # Récupération de la position actuelle du robot via l'odométrie
        pose = self.odometer_values()
        self.counter += 1
            
        # TP 3 :
        # Mise à jour de la carte avec les données du lidar et la position odométrique
        """ self.tiny_slam.update_map(self.lidar(), pose)
        if self.counter % 10 == 0:
            self.occupancy_grid.display_cv(pose, self.waypoints[self.current_waypoint_idx]) """

        # TP 4 :
        # Localisation
        score = self.tiny_slam.localise(self.lidar(), pose)
        corrected_pose = self.tiny_slam.get_corrected_pose(pose)
        print(f"Final Score = {score:.1f}")

        if -score > 16 or self.counter < 20:
            self.tiny_slam.update_map(self.lidar(), corrected_pose)

        if self.counter % 10 == 0:
            self.occupancy_grid.display_cv(corrected_pose, self.waypoints[self.current_waypoint_idx])
        
        return self.control_tp2()  

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path pla nning
        """
        pose = self.odometer_values()
        current_goal = self.waypoints[self.current_waypoint_idx]
        
        # Verifica se o waypoint foi alcançado
        distance_to_goal = np.linalg.norm(np.array(current_goal[:2]) - np.array(pose[:2]))
        if distance_to_goal < self.distance_threshold:
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            print(f"Waypoint {self.current_waypoint_idx} reached ! Next: {self.waypoints[self.current_waypoint_idx]}")

        command = potential_field_control(self.lidar(), pose, current_goal)
        return command
