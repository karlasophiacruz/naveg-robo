import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.pylab import rcParams


class treeNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []
        self.parent = None


class RRTAlgorithm():
    def __init__(self, start, goal, numInterations, grid, stepSize):
        self.randomTree = treeNode(start[0], start[1])
        self.goal = treeNode(goal[0], goal[1])
        self.nearestNode = None
        self.iterations = min(numInterations, 1000)
        self.grid = grid
        self.rho = stepSize
        self.path_distance = 0
        self.nearestDist = 10000
        self.numWaypoints = 0
        self.wayPoints = []

    # add the point to nearest node and add goal when reached
    def addChild(self, x, y):
        if self.goal.x == x:
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            tempNode = treeNode(x, y)
            self.nearestNode.children.append(tempNode)
            tempNode.parent = self.nearestNode

    # sample a random point within grid limits
    def samplePoint(self):
        x = random.randint(1, grid.shape[1] - 1)
        y = random.randint(1, grid.shape[0] - 1)

        return np.array([x, y])

    # steer a distance stepsize from start to end location
    def steer(self, start, end):
        offset = self.rho * self.unitVector(start, end)
        point = np.array([start.x + offset[0], start.y + offset[1]])

        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1] - 1
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[0] - 1

        return point

    # check if obstacle lies between the start node and end point of the edge
    def isInObstacle(self, start, end):
        u_hat = self.unitVector(start, end)
        testPoint = np.array([0, 0])

        for i in range(self.rho):
            testPoint[0] = start.x + i * u_hat[0]
            testPoint[1] = start.y + i * u_hat[1]

            if testPoint[0] >= grid.shape[1]:
                testPoint[0] = grid.shape[1] - 1
            if testPoint[1] >= grid.shape[0]:
                testPoint[1] = grid.shape[0] - 1

            if self.grid[round(testPoint[1].astype(np.int64)), round(testPoint[0].astype(np.int64))] == 1:
                return True

        return False

    # find unit vector from start to end
    def unitVector(self, start, end):
        v = np.array([end[0] - start.x, end[1] - start.y])

        return v / np.linalg.norm(v)

    # find the nearest node from a given unconnected point
    def findNearest(self, root, point):
        if not root:
            return

        dist = self.distance(root, point)

        if dist <= self.nearestDist:
            self.nearestDist = dist
            self.nearestNode = root

        for child in root.children:
            self.findNearest(child, point)
        pass

    # find euclidean distance between a node and an XY point
    def distance(self, node, point):
        return np.sqrt((node.x - point[0]) ** 2 + (node.y - point[1]) ** 2)

    # check if the goal has been reached within step size
    def goalFind(self, point):
        if self.distance(self.goal, point) <= self.rho:
            return True
        pass

    # reset nearestNode and nearestDist
    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000

    # trace the path from goal to start
    def traceRRTPath(self, goal):
        if not goal:
            print('Goal not found')
            return
        if goal.x == self.randomTree.x:
            return

        self.numWaypoints += 1

        currentPoint = np.array([goal.x, goal.y])
        self.wayPoints.insert(0, currentPoint)
        self.path_distance += self.rho
        self.traceRRTPath(goal.parent)


def on_key(event):
    if event.key == 'escape':
        plt.close()
        global stop_flag
        stop_flag = True


np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 14

grid = np.load('naveg-robo/atividade3/cspace.npy')
start = np.array([100, 150])
goal = np.array([750, 800])
numInterations = 500
stepSize = 50
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='r', fill=False)

fig, ax = plt.subplots(figsize=(10, 7))

plt.style.use('ggplot')

ax.imshow(grid, cmap='binary')
ax.plot(start[0], start[1], 'bo', markersize=10)
ax.plot(goal[0], goal[1], 'ro', markersize=8)
ax.add_patch(goalRegion)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('RRT Path Planning')
ax.grid(True)

rrt = RRTAlgorithm(start, goal, numInterations, grid, stepSize)

fig.canvas.mpl_connect('key_press_event', on_key)
stop_flag = False

for i in range(rrt.iterations):
    if stop_flag:
        break
    rrt.resetNearestValues()
    print(f'Iteration {i}')

    point = rrt.samplePoint()
    rrt.findNearest(rrt.randomTree, point)

    new = rrt.steer(rrt.nearestNode, point)
    bool = rrt.isInObstacle(rrt.nearestNode, new)

    if not bool:
        rrt.addChild(new[0], new[1])
        plt.pause(0.1)
        ax.plot([rrt.nearestNode.x, new[0]], [
                rrt.nearestNode.y, new[1]], 'go', linestyle='dotted')

        if rrt.goalFind(new):
            rrt.addChild(goal[0], goal[1])
            print(f'Goal reached at iteration {i}')
            break

if not stop_flag:
    rrt.traceRRTPath(rrt.goal)
    rrt.wayPoints.insert(0, start)

    print(f'Number of waypoints: {rrt.numWaypoints}')
    print(f'Path distance: {rrt.path_distance}')
    print(f'Waypoints: {rrt.wayPoints}')

    for i in range(len(rrt.wayPoints) - 1):
        ax.plot([rrt.wayPoints[i][0], rrt.wayPoints[i + 1][0]],
                [rrt.wayPoints[i][1], rrt.wayPoints[i + 1][1]], 'ro', linestyle='dotted')
        plt.pause(0.1)

    plt.tight_layout()
    plt.show()
