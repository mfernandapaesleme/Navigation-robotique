""" A set of robotics control functions """

import random
import numpy as np

last_turn_direction = None
stored_poses = []
wall_following = False
wall_following_direction = None

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance - detects walls in front and turns toward open space
    lidar: placebot object with lidar data
    """
    # Get lidar distance readings
    laser_dist = lidar.get_sensor_values()
    
    # Parameters
    safe_distance = 150  # Start reacting when obstacle is within this distance
    critical_distance = 50  # Distance for more aggressive avoidance
    default_speed = 0.3  # Normal forward speed
    medium_speed = 0.2  # Speed when approaching obstacles
    min_speed = 0.1  # Minimum speed when approaching obstacles
    
    # Initialize speeds
    speed = default_speed
    rotation_speed = 0.0


    # Get total number of lidar readings
    num_readings = len(laser_dist)
    if num_readings == 0:
        return {"forward": 0.0, "rotation": 0.0}
    
    """ front_indices = (60,120)
    left_indices = (0, 60)
    right_indices = (120, 180) """

    front_indices = (170,190)
    left_indices = (190,315)
    right_indices = (45, 170)
    
    # Get minimum distance in each region
    front_distances = [laser_dist[i] for i in front_indices]
    left_distances = [laser_dist[i] for i in left_indices]
    right_distances = [laser_dist[i] for i in right_indices]
    
    min_front = min(front_distances) if front_distances else float('inf')
    min_left = min(left_distances) if left_distances else float('inf')
    min_right = min(right_distances) if right_distances else float('inf')
    
    # Calculate speed based on front distance
    print(f"min_front: {min_front:.2f}, min_left: {min_left:.2f}, min_right: {min_right:.2f}")
    if min_front < safe_distance:
        speed = medium_speed
        if min_left > min_right:
            # More space to the left, turn left
            rotation_speed = 0.5 
        else:
            # More space to the right, turn right
            rotation_speed = -0.5
    if min_front < critical_distance:
        speed = min_speed
    else:
        speed = default_speed

    if (min_front < critical_distance) and (min_left < critical_distance) and (min_right < critical_distance):
        # If all directions are blocked, stop
        speed = -0.1
        rotation_speed = 0.6
    
    # Create command to return
    command = {
        "forward": speed,
        "rotation": rotation_speed
    }
    
    return command



def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    # Calcul du vecteur de distance
    distance_vector = goal_pose[:2] - current_pose[:2]
    distance = np.linalg.norm(distance_vector)
    
    # Paramètres
    K_goal = 1.0         # Coefficient attractif
    d_lim = 100         # Rayon du potentiel quadratique
    stop_threshold = 10  # Seuil d'arrêt
    dsafe = 500         # Distance de sécurité pour l'obstacle
    K_obs = 300         # Coefficient répulsif

    # Parameters
    safe_distance = 150  # Start reacting when obstacle is within this distance
    critical_distance = 50  # Distance for more aggressive avoidance
    default_speed = 0.1  # Normal forward speed
    medium_speed = 0.1 # Speed when approaching obstacles
    min_speed = 0.1  # Minimum speed when approaching obstacles
    
    # Initialize speeds
    forward_speed = default_speed
    rotation_speed = 0.0
    
    # Condition d'arrêt
    if distance < stop_threshold:
        print("Objectif atteint, arrêt")
        return {"forward": 0.0, "rotation": 0.0}
    
    # Gradient attractif
    if distance > d_lim:
        grad_att = (K_goal / distance) * distance_vector
    else:
        grad_att = (K_goal / d_lim) * distance_vector

    # Gradient répulsif
    grad_rep = np.array([0.0, 0.0])
    lidar_distances = lidar.get_sensor_values()
    lidar_angles = lidar.get_ray_angles()
    
    for i in range(len(lidar_distances)):
        d = lidar_distances[i]
        if d < dsafe:
            angle = lidar_angles[i]
            obs_position = np.array([np.cos(angle), np.sin(angle)]) * d  # en frame robot
            # print(f"Obstacle detected at distance {d:.2f} at angle {angle:.2f} rad")
            # Repulsive gradient formula
            rep = K_obs * ((1.0 / d) - (1.0 / dsafe)) / (d ** 3) * (obs_position)
            # Peso maior para obstáculos frontais
            weight = np.exp(-abs(angle))  # máximo para 0 rad (frente), decai pros lados
            grad_rep += weight * rep
    

    # Gradient total
    total_grad = grad_att + grad_rep
    if np.linalg.norm(total_grad) < 0.01:
        exploration_force = np.random.uniform(-0.5, 0.5, size=2)
        total_grad += exploration_force

    # Direction souhaitée
    grad_angle = np.arctan2(total_grad[1], total_grad[0])
    
    # Erreur d’orientation
    heading_error = grad_angle - current_pose[2]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    #front_indices = (140,220)
    #left_indices = (250,290)
    #right_indices = (70, 110)
    front_indices = (170,190)
    left_indices = (190,315)
    right_indices = (45, 170)
    
    # Get minimum distance in each region
    front_distances = [lidar_distances[i] for i in front_indices]
    left_distances = [lidar_distances[i] for i in left_indices]
    right_distances = [lidar_distances[i] for i in right_indices]
    
    min_front = min(front_distances) if front_distances else float('inf')
    min_left = min(left_distances) if left_distances else float('inf')
    min_right = min(right_distances) if right_distances else float('inf')

    # print(f"min_front: {min_front:.2f}") #min_left: {min_left:.2f}, min_right: {min_right:.2f}")

    # heading_error: radianos (-pi a pi)
    # min_distance: em metros

    if (min_front < critical_distance) and (min_left < critical_distance) and (min_right < critical_distance):
        # If all directions are blocked, stop
        forward_speed = -0.1
        rotation_speed = 0.6

    # Variações de velocidade baseadas no erro de orientação
    if abs(heading_error) > 0.5:
        forward_speed = 0.0
    elif abs(heading_error) > 0.1:
        forward_speed = medium_speed 
    else:
        if min_front < safe_distance:
            forward_speed = medium_speed  # anda devagar perto da parede
        elif min_front < critical_distance:
            forward_speed = min_speed
        else:
            forward_speed = default_speed  # anda rápido se longe da parede e bem alinhado

    
    """ # Variação do ganho de rotação baseada na distância ao obstáculo
    if min_front < critical_distance:
        k_rot = 2.0  # gira mais forte se perto
    else:
        k_rot = 0.2  # gira normal

    rotation_speed = k_rot * heading_error
 """
    wall_following = False
    wall_following_direction = None
    
    # Wall following
    if min_front < critical_distance: 
        if 0.01 < heading_error < 0.1 or -0.1 < heading_error < -0.01:
            print("perto da parede e meio alinhado girando um pouco")
            k_rot = 2.5 #sipa da p aumentar

        elif 0 < heading_error < 0.01 or -0.01 < heading_error < 0:
            print("perto da parede e muito alinhado girando muito")
            k_rot = 100
            forward_speed = default_speed if min_front > critical_distance/2 else 0.1
        else:
            k_rot = 2.5
        
        if is_stuck(current_pose):
            print("estou preso, ativar wall following")
            wall_following = True
            if wall_following_direction is None or wall_following_direction == "right":
                wall_following_direction = "left"
            else:
                wall_following_direction = "right"

            while wall_following:
                if wall_following_direction == "left":
                    rotation_speed = 0.5
                    forward_speed = 0.1
                    if min_left > critical_distance:
                        wall_following = False
                else:
                    rotation_speed = -0.5
                    forward_speed = 0.1
                    if min_right > critical_distance:
                        wall_following = False
        
    else:
        k_rot = 0.2  # gira normal
    
    rotation_speed = k_rot * heading_error

    if rotation_speed > 0:
        last_turn_direction = "right"
    elif rotation_speed < 0:
        last_turn_direction = "left"

    """ if is_stuck(current_pose):
        print("estou preso")
        if last_turn_direction is None or last_turn_direction == "right":
            rotation_speed = -0.5
            last_turn_direction = "left"
        else:
            rotation_speed = 0.5
            last_turn_direction = "right"
 """
    """ if min_front < critical_distance and 0 < heading_error < 0.01 :
        print("Muito perto e errado, gira para a esquerda")
        rotation_speed = 0.5 

    if min_front < critical_distance and 0 > heading_error > -0.01 :
        print("Muito perto e errado, gira para a direita")
        rotation_speed = -0.5 """


    # Normalisation des vitesses
    forward_speed = np.clip(forward_speed, -1.0, 1.0)
    rotation_speed = np.clip(rotation_speed, -1.0, 1.0)

    command = {
        "forward": forward_speed,
        "rotation": rotation_speed
    }

    """ print(f"pose: {current_pose}")
    print(f"heading_error: {heading_error}, k_rot: {k_rot}")
    print(f"Distance: {distance:.2f}, Gradient att: {grad_att}, Gradient rep: {grad_rep}")
    print(f"Command: forward={forward_speed:.2f}, rotation={rotation_speed:.2f}") """

    return command

def is_stuck(corrected_pose, std_thresh=15.0, max_len=300):
    # Armazena apenas as coordenadas x, y
    stored_poses.append(corrected_pose[:2])

    # Limita o tamanho da lista de poses
    if len(stored_poses) > max_len:
        stored_poses.pop(0)

    if len(stored_poses) < max_len:
        return False

    # Extrai listas separadas de coordenadas x e y
    x_coords, y_coords = zip(*stored_poses)
    
    std_x = np.std(x_coords)
    std_y = np.std(y_coords)
    
    # Verifica se os desvios padrão são menores que o limiar
    return std_x < std_thresh and std_y < std_thresh