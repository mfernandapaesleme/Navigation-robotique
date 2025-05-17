""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        self.odom_pose_init = None

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        score = 0

        # Récupération des données du lidar
        ranges = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        max_lidar_range = lidar.max_range

        # Filter out max distance measurements (not obstacles)
        valid_mask = (ranges > 0) & (ranges < max_lidar_range)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]

        x, y, theta = pose

        # Compute the absolute angles of the rays
        world_angles = theta + angles
        # Compute the end points of the rays (polaires) in world coordinates (cartesian)
        world_x = x + ranges * np.cos(theta + world_angles)
        world_y = y + ranges * np.sin(theta + world_angles)

        world_end_x, world_end_y = self.grid.conv_world_to_map(world_x, world_y)

        # Keep only values inside the map
        isValid = (world_end_x < self.grid.x_max_map) & (world_end_x >= 0) & (world_end_y < self.grid.y_max_map) & (world_end_y >= 0)

        valid_world_end_x = world_end_x[isValid]
        valid_world_end_y = world_end_y[isValid]

        # Convert the world end points to integer indices
        valid_world_end_x = valid_world_end_x.astype(int)
        valid_world_end_y = valid_world_end_y.astype(int)

        log_probs = self.grid.occupancy_map[valid_world_end_x, valid_world_end_y] # Get the log-odds values from the occupancy grid
        # Compute the log-odds score  
        score = np.sum(log_probs)  
       
        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        corrected_pose = odom_pose

        if self.odom_pose_init is None: 
            self.odom_pose_init = odom_pose.copy()

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        
        # Get displacements from initial pose
        dx = odom_pose[0] - self.odom_pose_init[0]
        dy = odom_pose[1] - self.odom_pose_init[1]
        dtheta = odom_pose[2] - self.odom_pose_init[2]
        
        # Apply transformation
        corrected_x = odom_pose_ref[0] + dx * np.cos(odom_pose_ref[2]) - dy * np.sin(odom_pose_ref[2])
        corrected_y = odom_pose_ref[1] + dx * np.sin(odom_pose_ref[2]) + dy * np.cos(odom_pose_ref[2])
        corrected_theta = odom_pose_ref[2] + dtheta
        
        # Normalize angle
        corrected_theta = np.arctan2(np.sin(corrected_theta), np.cos(corrected_theta))
        
        # Normalize angle to [-pi, pi]
        corrected_theta = np.arctan2(np.sin(corrected_theta), np.cos(corrected_theta))
        """ print(f"Odom: {odom_pose} | Odom_init: {self.odom_pose_init}")
        print(f"Displacement: dx={dx:.2f}, dy={dy:.2f}, dtheta={dtheta:.2f}")
        print(f"Reference: x={odom_pose_ref[0]:.2f}, y={odom_pose_ref[1]:.2f}, theta={odom_pose_ref[2]:.2f}")
        print(f"Corrected: x={corrected_x:.2f}, y={corrected_y:.2f}, theta={corrected_theta:.2f}") """

        corrected_pose = np.array([corrected_x, corrected_y, corrected_theta])
        return corrected_pose
    

    def localise(self, lidar, raw_odom_pose, max_iter=30):
        """Constrained localization with stability checks"""
        current_pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_pose)
        
        # Only attempt localization if current score is poor
        if best_score > 50:  # Good enough score threshold
            return best_score
            
        best_ref = self.odom_pose_ref.copy()
        
        # Tight search constraints (meters, radians)
        search_sigma = [1, 1, 0.1]  # x,y,theta
        
        for _ in range(max_iter):
            offset = np.random.normal(0, search_sigma, 3)
            test_ref = best_ref + offset
            test_pose = self.get_corrected_pose(raw_odom_pose, test_ref)
            test_score = self._score(lidar, test_pose)
            
            # Only accept significant improvements
            if test_score > best_score + 10:  
                best_score = test_score
                best_ref = test_ref
        
        # Only update reference if major improvement
        if best_score > 50:  # Absolute quality threshold
            self.odom_pose_ref = best_ref
        
        return best_score
    
    def conv_donnes(self, ranges, ray_angles, pose, isWall):
        # Position du robot dans le repère absolu
        robot_x, robot_y, robot_theta = pose
                
        # Points détectés en coordonnées polaires -> coordonnées cartésiennes dans le repère robot
        # Calcul de l'angle absolu du rayon
        ray_angle_world = robot_theta + ray_angles
                
        # Conversion en coordonnées cartésiennes dans le repère robot
        if isWall == True:
            ray_x_robot = (ranges + 10) * np.cos(ray_angle_world)
            ray_y_robot = (ranges + 10) * np.sin(ray_angle_world)
        else:
            ray_x_robot = ranges * np.cos(ray_angle_world)
            ray_y_robot = ranges * np.sin(ray_angle_world)
                
        # Conversion en coordonnées cartésiennes dans le repère absolu
        ray_x_world = robot_x + ray_x_robot
        ray_y_world = robot_y + ray_y_robot

        return ray_x_world, ray_y_world



    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        
        """  # Bayesian update probabilities
        p_free = 0.4  # Probability for free space (lower than 0.5)
        p_occ = 0.7   # Probability for occupied space (higher than 0.5)

        # Valeurs à ajouter à la grille d'occupation logarithmique
        val_free = np.log(p_free / (1 - p_free))
        val_occ = np.log(p_occ / (1 - p_occ)) """ 
        val_free = -1
        val_occ = 1

        # Récupération des données du lidar
        ranges = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        
        # Stockage des points pour mise à jour de la carte
        points_x, points_y = self.conv_donnes(ranges, ray_angles, pose, False)

        # Mise à jour de la carte pour les obstacles (points détectés)
        self.grid.add_map_points(points_x, points_y, val_occ)

        # Stockage des points pour mise à jour de la carte
        # points_x, points_y = self.conv_donnes(ranges, ray_angles, pose, True)
    
        # Mise à jour de la carte pour les zones libres (trajectoire du rayon)
        for xi, yi in zip(points_x, points_y):
            self.grid.add_value_along_line(pose[0], pose[1], xi, yi, val_free)
        
        # Clip the occupancy grid values to prevent overflow
        max_log_odds = 40  # Maximum log-odds value
        min_log_odds = -40  # Minimum log-odds value
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, min_log_odds, max_log_odds)

    """ def compute(self):
        Useless function, just for the exercise on using the profiler
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y]) """
