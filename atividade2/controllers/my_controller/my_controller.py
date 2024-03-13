import math
import numpy as np
from controller import Lidar, Motor, Supervisor

TIME_STEP = 32
BASE_SPEED = 1
BUG_DISTANCE_THRESHOLD = 0.2
MAX_SPEED = 6.28
WHEEL_DISTANCE = 0.160
WHEEL_RADIUS = 0.033

linear_velocity = 2 * WHEEL_RADIUS * MAX_SPEED

def gaussian(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(
        -((x - mu) * (x - mu)) / (2 * sigma * sigma)
    )

def set_velocity(current_to_goal_angle, current_rotation):
    y_angle = math.sin(current_to_goal_angle - current_rotation)
    x_angle = math.cos(current_to_goal_angle - current_rotation)

    angular_velocity = math.atan2(y_angle, x_angle)

    v_r = min(
        MAX_SPEED,
        (linear_velocity + (angular_velocity * WHEEL_DISTANCE)) / (2 * WHEEL_RADIUS),
    )
    v_l = min(
        MAX_SPEED,
        (linear_velocity - (angular_velocity * WHEEL_DISTANCE)) / (2 * WHEEL_RADIUS),
    )

    return v_r, v_l

def move_to_goal():
    global goal_reached

    if not goal_reached:
        current_pos = turtlebot_node.getPosition()
        goal_pos = duck_node.getPosition()

        ttbot_to_duck_angle = math.atan2(
            goal_pos[1] - current_pos[1], goal_pos[0] - current_pos[0]
        )
        ttbot_rotation = math.atan2(
            turtlebot_node.getOrientation()[3], turtlebot_node.getOrientation()[0]
        )

        v_r, v_l = set_velocity(ttbot_to_duck_angle, ttbot_rotation)

        left_motor.setVelocity(v_l)
        right_motor.setVelocity(v_r)

        # Verificar se o robô está próximo o suficiente do objetivo
        diff_position = [duck - turtle for duck, turtle in zip(goal_pos, current_pos)]
        distance_to_goal = math.sqrt((diff_position[0]) ** 2 + (diff_position[1]) ** 2)

        if distance_to_goal < 0.15:
            goal_reached = True
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)

def follow_wall():
    global braitenberg_coefficients

    left_speed = BASE_SPEED
    right_speed = BASE_SPEED

    lidar_values = lidar.getRangeImage()

    for i in range(int(0.25 * lidar_width), int(0.5 * lidar_width)):
        j = lidar_width - i - 1
        k = i - int(0.25 * lidar_width)
        if (
            lidar_values[i] != float("inf")
            and not math.isnan(lidar_values[i])
            and lidar_values[j] != float("inf")
            and not math.isnan(lidar_values[j])
        ):
            left_speed += braitenberg_coefficients[k] * (
                (1.0 - lidar_values[i] / lidar_max_range)
                - (1.0 - lidar_values[j] / lidar_max_range)
            )
            right_speed += braitenberg_coefficients[k] * (
                (1.0 - lidar_values[j] / lidar_max_range)
                - (1.0 - lidar_values[i] / lidar_max_range)
            )

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

def switch_mode(distance_to_obstacle):
    global mode, goal_reached

    if distance_to_obstacle <= BUG_DISTANCE_THRESHOLD and mode == "move_to_goal":
        mode = "follow_wall"
    elif distance_to_obstacle > BUG_DISTANCE_THRESHOLD and mode == "follow_wall":
        mode = "move_to_goal"
        goal_reached = False

supervisor = Supervisor()

lidar = supervisor.getDevice("LDS-01")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

lidar_main_motor = supervisor.getDevice("LDS-01_main_motor")
lidar_secondary_motor = supervisor.getDevice("LDS-01_secondary_motor")
lidar_main_motor.setPosition(float("inf"))
lidar_secondary_motor.setPosition(float("inf"))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

right_motor = supervisor.getDevice("right wheel motor")
left_motor = supervisor.getDevice("left wheel motor")
right_motor.setPosition(float("inf"))
left_motor.setPosition(float("inf"))
right_motor.setVelocity(0.0)
left_motor.setVelocity(0.0)

lidar_width = lidar.getHorizontalResolution()
lidar_max_range = lidar.getMaxRange()

braitenberg_coefficients = np.zeros(lidar_width)
for i in range(lidar_width):
    braitenberg_coefficients[i] = 6 * gaussian(i, lidar_width / 4, lidar_width / 12)

duck_node = supervisor.getFromDef("duck")
turtlebot_node = supervisor.getFromDef("turtlebot")

mode = "move_to_goal"
goal_reached = False

while supervisor.step(TIME_STEP) != -1:
    lidar_values = lidar.getRangeImage()
    min_distance_to_obstacle = min(lidar_values)
    print(min_distance_to_obstacle)
    
    if mode == "move_to_goal":
        move_to_goal()
        switch_mode(min_distance_to_obstacle)
    elif mode == "follow_wall":
        follow_wall()
        if min_distance_to_obstacle > BUG_DISTANCE_THRESHOLD:
            mode = "move_to_goal"
            goal_reached = False