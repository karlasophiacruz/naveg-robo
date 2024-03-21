import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
OBSTACLE_RADIUS = 4


class AttractivePotential:
    def __init__(self, goal_point):
        self.goal_point = goal_point
        self.KP = 5
        self.constant = 0.5 * self.KP

    def compute(self, position):
        return self.constant * np.linalg.norm(position - self.goal_point) ** 2


class RepulsivePotential:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.ETA = 50
        self.constant = 0.5 * self.ETA

    def distance_from_circle(self, position, center, radius):
        return np.linalg.norm(position - center) - radius

    def compute(self, position):
        potential = 0

        for obstacle in self.obstacles:
            obstacle_position = np.array(obstacle)
            distance = self.distance_from_circle(
                position, obstacle_position, OBSTACLE_RADIUS)

            if distance != 0:
                potential += self.constant * \
                    (1.0 / distance - 1.0 / self.ETA) ** 2

        return potential

    def compute_obstacle(self, X, Y):
        U_rep = np.zeros_like(X)

        for obstacle in self.obstacles:
            center = obstacle
            distance_to_center = np.linalg.norm(
                np.stack((X - center[0], Y - center[1]), axis=-1), axis=-1)

            U_rep[distance_to_center <= OBSTACLE_RADIUS] = self.constant

        return U_rep


class CombinedPotential:
    def __init__(self, attractive_field, repulsive_field):
        self.attractive_field = attractive_field
        self.repulsive_field = repulsive_field

    def compute(self, position):
        return self.attractive_field.compute(position) + self.repulsive_field.compute(position)


def gradient(start_point, potential_field, learning_rate=0.1, max_iterations=1000, tolerance=1e-5):
    position = np.array(start_point, dtype=float)
    trajectory = [position.copy()]

    for _ in range(max_iterations):
        grad_x = (potential_field.compute(
            position + [tolerance, 0]) - potential_field.compute(position - [tolerance, 0])) / (2.0 * tolerance)

        grad_y = (potential_field.compute(
            position + [0, tolerance]) - potential_field.compute(position - [0, tolerance])) / (2.0 * tolerance)

        grad = np.array([grad_x, grad_y], dtype=float)
        grad /= np.linalg.norm(grad) + 1e-8

        position -= learning_rate * grad
        trajectory.append(position.copy())

        if np.linalg.norm(grad) < tolerance:
            break
    return np.array(trajectory)


def plot_3d_surface(fig, i, X, Y, Z):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_zlabel('Potencial')


start_point = (7, 2)
goal_point = (37, 38)
obstacles = [(12, 38), (14, 20), (20, 4), (35, 32), (25, 25), (5, 7)]

x = np.linspace(0, 40, 100)
y = np.linspace(0, 40, 100)
X, Y = np.meshgrid(x, y)

attractive_field = AttractivePotential(goal_point)
repulsive_field = RepulsivePotential(obstacles)
potential_field = CombinedPotential(attractive_field, repulsive_field)
trajectory = gradient(start_point, potential_field)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax in [axes[0, 0], axes[0, 1]]:
    for obstacle in obstacles:
        ax.add_patch(plt.Circle(obstacle, OBSTACLE_RADIUS, color='orange'))
    ax.plot(start_point[0], start_point[1], 'bo',
            markersize=8, label='Ponto Inicial')
    ax.plot(goal_point[0], goal_point[1], 'ro',
            markersize=8, label='Ponto Final')
axes[0, 1].plot(trajectory[:, 0], trajectory[:, 1], 'g-', label='Trajetória')

Z_attractive = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        position = np.array([X[i, j], Y[i, j]])
        Z_attractive[i, j] = attractive_field.compute(position)
plot_3d_surface(fig, 3, X, Y, Z_attractive)

Z_repulsive = repulsive_field.compute_obstacle(X=X, Y=Y)
plot_3d_surface(fig, 4, X, Y, Z_repulsive)

titles = ['Mapa Utilizado', 'Trajetória percorrida',
          'Campo Potencial Atrativo', 'Campo Potencial Repulsivo']
for ax in axes.flat:
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.set_title(titles.pop(0))

plt.tight_layout()
plt.show()
