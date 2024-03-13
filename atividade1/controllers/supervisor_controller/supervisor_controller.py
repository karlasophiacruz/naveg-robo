from controller import Supervisor
import math

TIME_STEP = 32
MAX_SPEED = 6.28
WHEEL_DISTANCE = 0.160
WHEEL_RADIUS = 0.033

linear_velocity = 2 * WHEEL_RADIUS * MAX_SPEED

supervisor = Supervisor()

turtlebot_node = supervisor.getFromDef('turtlebot')
duck_node = supervisor.getFromDef('duck')

leftMotor = supervisor.getDevice('left wheel motor')
rightMotor = supervisor.getDevice('right wheel motor')

leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

def set_velocity():
    # Calcula o angulo entre o pato e o robo, considerando a rotação do robô
    ttbot_to_duck_angle = math.atan2(duck_pos[1] - turtle_pos[1], duck_pos[0] - turtle_pos[0])
    ttbot_rotation = math.atan2(turtlebot_node.getOrientation()[3], turtlebot_node.getOrientation()[0])
    
    y_angle = math.sin(ttbot_to_duck_angle - ttbot_rotation)
    x_angle = math.cos(ttbot_to_duck_angle - ttbot_rotation)
    
    angular_velocity = math.atan2(y_angle, x_angle)
    
    # Calcula a velocidade das rodas considerando as formulas diferenciais
    v_r = min(MAX_SPEED, (linear_velocity + (angular_velocity * WHEEL_DISTANCE)) / (2 * WHEEL_RADIUS))
    v_l = min(MAX_SPEED, (linear_velocity - (angular_velocity * WHEEL_DISTANCE)) / (2 * WHEEL_RADIUS))

    return v_r, v_l
    
while supervisor.step(TIME_STEP) != -1:
    duck_pos = duck_node.getPosition()
    turtle_pos = turtlebot_node.getPosition()
    
    diff_position = [duck - turtle for duck, turtle in zip(duck_pos, turtle_pos)] 
    distance = math.sqrt((diff_position[0])**2 + (diff_position[1])**2)
    
    # Checa se já chegou no pato
    if distance < 0.15:
        v_l = v_r = 0
        
    else:
        v_r, v_l = set_velocity()

    
    leftMotor.setVelocity(v_l)
    rightMotor.setVelocity(v_r)
