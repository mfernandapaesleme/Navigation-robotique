"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
from matplotlib import pyplot as plt
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
            [-400, -520, 0],
            [-200, -450, 0],     # Ponto 4: trás
            [-200, -200, 0],
            [-200, -80, 0],
            [-400, -100, 0],
            [-400, 0, 0],
            [-200, 10, 0],
            [-600, 10, 0],
            [-780, -100, 0],
            [-780, -150, 0],
            [-700, -150, 0],
            [-700, -200, 0],
            [-690, -200, 0],
            [-750, -100, 0],
            [-800, -150, 0],
            [-780, -250, 0],
            [-780, -350, 0],
            [-900, -380, 0],
            [-780, -350, 0], 
            [-780, -260, 0],
            [-900, -250, 0],
            [-950, -150, 0],
            [-950, -50, 0],
            [-950, -50, 0]
                                  
        ]
        self.current_waypoint_idx = 0
        self.distance_threshold = 50 
        self.seuil = 200 
        self.goal_reached = False

        self.path_found = False
        self.path = []
        self.paths = []


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
        self.corrected_pose = corrected_pose
        # print(f"Final Score = {score:.1f}")

        if self.counter < 40:
            self.tiny_slam.update_map(self.lidar(),self.odometer_values())            
            
            return {"forward": 0,
                    "rotation": 0}
        
        # Start moving and only update the map if the robot is localized
        elif self.goal_reached == False:
            if score > self.seuil:
                self.tiny_slam.update_map(self.lidar(), corrected_pose)

            if(self.counter == 20):
                self.seuil = 400

            if self.counter % 10 == 0:
                self.occupancy_grid.display_cv(corrected_pose, self.waypoints[self.current_waypoint_idx])

            return self.control_tp2()
        
        # TP 5 :
        # After 6000 steps, start path planning
        else:
            """ # Atualiza mapa se localização for boa
            if score > self.seuil:
                self.tiny_slam.update_map(self.lidar(), corrected_pose) """
            
            # PLANEJAMENTO DO CAMINHO (apenas uma vez)
            if not self.path_found:
                print("Iniciando planejamento de caminho...")
                
                # Processa a grade de ocupação para planejamento
                # 1. Aplica threshold: valores > 35 viram obstáculos
                processed_grid = self.planner.occupancy_grid_threshold(
                    self.tiny_slam.grid.occupancy_map, threshold=35)
                
                # 2. Dilata obstáculos para margem de segurança
                processed_grid = self.planner.occupancy_grid_dilate(
                    processed_grid, radius=7)
                
                # 3. Atualiza a grade do planner
                self.planner.grid.occupancy_map = processed_grid
                
                # 4. Planeja caminho da posição atual para origem (0,0,0)
                path_cells = self.planner.plan(corrected_pose, np.array([0, 0, 0]))
                
                if path_cells is not None:
                    # Converte caminho de células para coordenadas do mundo
                    self.paths = []
                    for cell in path_cells:
                        world_coord = self.planner.grid.conv_map_to_world(cell[0], cell[1])
                        self.paths.append(world_coord)
                    
                    print(f"Caminho encontrado com {len(self.paths)} pontos!")
                    self.path_found = True
                    
                    # Salva a posição atual corrigida para controle
                    self.corrected_pose = corrected_pose
                else:
                    print("Nenhum caminho encontrado!")
                    return {"forward": 0, "rotation": 0}
                

            # SEGUIMENTO DO CAMINHO
            if self.path_found and len(self.paths) > 0:
                # Converter paths em array NumPy antes de exibir
                traj_array = np.array([[p[0] for p in self.paths],[p[1] for p in self.paths]])
                objective = self.paths[-1] # Último ponto é o objetivo final

                # Exibir mapa e trajetória
                if self.counter % 10 == 0:
                    self.occupancy_grid.display_cv(corrected_pose, objective, traj_array)
            
                # Atualiza posição corrigida
                self.corrected_pose = corrected_pose
                
                # Encontra o ponto mais próximo no caminho
                distances = [np.linalg.norm(np.array(point) - corrected_pose[:2]) 
                           for point in self.paths]
                closest_idx = np.argmin(distances)
                
                # Define ponto objetivo alguns passos à frente no caminho
                lookahead = 10  # Quantos pontos olhar à frente
                target_idx = min(closest_idx + lookahead, len(self.paths) - 1)
                goal_point = self.paths[target_idx]
                
                # Verifica se chegou ao destino final
                distance_to_end = np.linalg.norm(np.array(self.paths[-1]) - corrected_pose[:2])
                if distance_to_end < 30:  # 30 pixels de tolerância
                    print("Chegou ao destino final!")
                    return {"forward": 0, "rotation": 0}
                
                # Controle para seguir o caminho
                goal_3d = [goal_point[0], goal_point[1], 0]
                command = potential_field_control(self.lidar(), corrected_pose, goal_3d)
                return command

            
            # Se não há caminho, para
            return {"forward": 0, "rotation": 0}
        

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command
    
    def control_stop(self):
        """
        Control function for TP1_2
        Control funtion with minimal random motion
        """
        speed = 0.0
        rotation_speed = 0.0

        command = {"forward": speed,
                   "rotation": rotation_speed}
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        # for tp4
        corrected_pose = self.corrected_pose
        current_goal = self.waypoints[self.current_waypoint_idx]
        
        # Verifica se o waypoint foi alcançado
        distance_to_goal = np.linalg.norm(np.array(current_goal[:2]) - np.array(corrected_pose[:2]))
        if distance_to_goal < self.distance_threshold:
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            print(f"Waypoint {self.current_waypoint_idx} reached ! Next: {self.waypoints[self.current_waypoint_idx]}")
        
        if self.current_waypoint_idx >= len(self.waypoints)-1:
            print("All waypoints reached !")
            self.goal_reached = True
        command = potential_field_control(self.lidar(), corrected_pose, current_goal)
        return command
