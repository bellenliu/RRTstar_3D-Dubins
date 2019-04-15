import random
import math
import copy
import numpy as np
import time
import dubins_3Dpath_planning
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

show_animation = True
show_path = True
fig = plt.figure()

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 goalSampleRate=35, maxIter=2500):
        """
        Setting Parameter

        start:Start Position [x,y,z,psi,gamma]
        goal:Goal Position [x,y,z,psi,gamma]
        obstacleList:obstacle Positions [[x,y,z,size],...] -----> Spherical Obstacles
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start[0], start[1], start[2], start[3], start[4])
        self.end = Node(goal[0], goal[1], goal[2], goal[3], goal[4])
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        
        for i in range(self.maxIter):
            SteerCheck = False
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            while(not SteerCheck):                
                try:
                    newNode = self.steer(rnd, nind)
                    SteerCheck = True
                except:
                    rnd = self.get_random_point()
                    nind = self.GetNearestListIndex(self.nodeList, rnd)

            if self.CollisionCheck(newNode, self.obstacleList):
                nearinds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearinds)
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)

            if animation:
#            if animation and i % 5 == 0:
                self.DrawGraph(rnd=rnd, itr=i, animation=animation)

        # generate coruse
        lastIndex = self.get_best_last_index()
        #  print(lastIndex)

        if lastIndex is None:
            return None

        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if not nearinds:
            return newNode

        dlist = []
        for i in nearinds:
            try:
                tNode = self.steer(newNode, i)
                if self.CollisionCheck(tNode, self.obstacleList):
                    dlist.append(tNode.cost)     
                else:
                    dlist.append(float("inf"))
            except:
                dlist.append(float("inf"))
        
        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode = self.steer(newNode, minind)

        return newNode

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def steer(self, rnd, nind):
        #  print(rnd)
        Rmin = 1.0

        nearestNode = self.nodeList[nind]

        px, py, pz, ppsi, pgamma, clen, mode = dubins_3Dpath_planning.dubins_3Dpath_planning(
            nearestNode.x, nearestNode.y, nearestNode.z, nearestNode.psi, nearestNode.gamma, 
            rnd.x, rnd.y, rnd.z, rnd.psi, rnd.gamma, Rmin)

        newNode = copy.deepcopy(nearestNode)
        newNode.x = px[-1]
        newNode.y = py[-1]
        newNode.z = pz[-1]
        newNode.psi = ppsi[-1]    
        newNode.gamma = pgamma[-1]

        newNode.path_x = px
        newNode.path_y = py
        newNode.path_z = pz
        newNode.path_psi = ppsi
        newNode.path_gamma = pgamma
        newNode.cost += clen
        newNode.parent = nind

        return newNode

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand),
                   random.uniform(-math.pi, math.pi),
                   random.uniform(-0.5*math.pi, 0.5*math.pi)
                   ]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y, self.end.z ,self.end.psi, self.end.gamma]

        node = Node(rnd[0], rnd[1], rnd[2], rnd[3], rnd[4])

        return node

    def get_best_last_index(self):
        #  print("get_best_last_index")
        
        # Angle Threshold
        PSITH = np.deg2rad(1.0)
        GAMMATH = np.deg2rad(1.0)
        # Distance Threshold
        XYZTH = 0.75

        goalinds = []
        for (i, node) in enumerate(self.nodeList):
            if self.calc_dist_to_goal(node.x, node.y, node.z) <= XYZTH:
                goalinds.append(i)

        # angle check
        fgoalinds = []
        for i in goalinds:
            if abs(self.nodeList[i].psi - self.end.psi) <= PSITH and \
               abs(self.nodeList[i].gamma - self.end.gamma) <= GAMMATH:
                fgoalinds.append(i)

        if not fgoalinds:
            return None

        mincost = min([self.nodeList[i].cost for i in fgoalinds])
        for i in fgoalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y, self.end.z]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            for (ix, iy, iz) in zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_z)):
                path.append([ix, iy, iz])
            #  path.append([node.x, node.y])
            goalind = node.parent
        path.append([self.start.x, self.start.y, self.start.z])
        return path

    def calc_dist_to_goal(self, x, y, z):
        return np.linalg.norm([x - self.end.x, y - self.end.y, z - self.end.z])

    def find_near_nodes(self, newNode):
        nnode = len(self.nodeList)
        d = 3.0  #Search space dimension 
        r = 50.0 * ((math.log(nnode) / nnode))**(1.0/d)
        #  r = self.expandDis * 5.0
        dlist = [(node.x - newNode.x) ** 2 +
                 (node.y - newNode.y) ** 2 +
                 (node.z - newNode.z) ** 2 +
                 (node.psi - newNode.psi) ** 2 +
                 (node.gamma - newNode.gamma) ** 2
                 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def rewire(self, newNode, nearinds):

        nnode = len(self.nodeList)

        for i in nearinds:
            nearNode = self.nodeList[i]
            try:
                tNode = self.steer(nearNode, nnode - 1)                
                obstacleOK = self.CollisionCheck(tNode, self.obstacleList)
                imporveCost = nearNode.cost > tNode.cost
    
                if obstacleOK and imporveCost:
                    #  print("rewire")
                    self.nodeList[i] = tNode
            except:
                continue
            
    def DrawGraph(self, rnd=None, itr=None, path=None, animation=False):  # pragma: no cover
        """
        Draw Graph
        """
        plt.clf()        
        ax = plt.axes(projection='3d')
        if rnd is not None and itr is not None:
             
            fig.suptitle('Iteration %i' %(itr+1), fontsize=20)
            ax.scatter(rnd.x, rnd.y, rnd.z, c="k", marker="^")
         
        if animation is True:
            for node in self.nodeList:
                if node.parent is not None:
                    ax.plot3D(node.path_x, node.path_y, node.path_z, "-g")
            
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        for (ox, oy, oz, size) in self.obstacleList:
            x = ox + size * np.outer(np.cos(u), np.sin(v))
            y = oy + size * np.outer(np.sin(u), np.sin(v))
            z = oz + size * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='r')

#        dubins_path_planning.plot_arrow(
#            self.start.x, self.start.y, self.start.yaw)
#        dubins_path_planning.plot_arrow(
#            self.end.x, self.end.y, self.end.yaw)
        if path is not None:
            ax.plot3D([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-k')
        plt.xlim(-3, 16)
        plt.ylim(-3, 16)
        ax.set_zlim3d(bottom=-3, top=16)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
            
#        ax.set_xlim3d(bottom=-3, top=16)
#        ax.set_ylim3d(bottom=-3, top=16)
#        ax.set_zlim3d(bottom=-3, top=16)
        
        
        plt.pause(0.01)
        #  input()

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd.x) ** 2 +
                 (node.y - rnd.y) ** 2 +
                 (node.z - rnd.z) ** 2 +
                 (node.psi - rnd.psi) ** 2 + 
                 (node.gamma - rnd.gamma) ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind

    def CollisionCheck(self, node, obstacleList):
        SafetyTh = 0.5
        for (ox, oy, oz, size) in obstacleList:
            for (ix, iy, iz) in zip(node.path_x, node.path_y, node.path_z):
                dx = ox - ix
                dy = oy - iy
                dz = oz - iz
                d = dx * dx + dy * dy + dz * dz
                if d <= size ** 2 + SafetyTh:
                    return False  # collision

        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, z, psi, gamma):
        self.x = x
        self.y = y
        self.z = z
        self.psi = psi
        self.gamma = gamma
        self.path_x = []
        self.path_y = []
        self.path_z = []
        self.path_psi = []
        self.path_gamma = []
        self.cost = 0.0
        self.parent = None


def main():
    print("Start RRT star with 3D Dubins Planning")

    # ====Search Path with RRT====
#    obstacleList = [
#        (5, 5, 1),
#        (3, 6, 2),
#        (3, 8, 2),
#        (3, 10, 2),
#        (7, 5, 2),
#        (9, 5, 2)
#    ]  # [x,y,size(radius)]
    
    obstacleList = [
        (3, 3, 1, 1.5),
#        (9, 8, 4, 1.5),
        (4, 8, 3, 2),
        (10, 4, 2, 1),
    ]  # [x,y,z,size(radius)]
    
#    obstacleList = [
#        (3, 0, 1.5),
#        (3, 4, 1.5),
#        (3, 10, 2),
#        (8, 8, 2),
#        (8, 4, 1),
#        (10, 6, 1),
#    ]  # [x,y,z,size(radius)]

    # Set Initial parameters
    start = [0.0, 0.0, 0.0, np.deg2rad(0.0), np.deg2rad(0.0)]
    goal = [12.0, 12.0, 6.0 ,np.deg2rad(-30.0), np.deg2rad(0.0)]

    t1 = time.time()
    rrt = RRT(start, goal, randArea=[-2.0, 15.0], obstacleList=obstacleList)    
    path = rrt.Planning(animation=show_animation)
    if (path is not None):
        print("Path found")
        rrt.DrawGraph(path=path, animation=show_animation)
    else:
        print("Path not found")
    t2 = time.time()
    print("Execution Time(in sec) -----> ",t2-t1)
    
    if (path is not None):
    # Draw final path
        if show_path:
            plt.figure(2)
            ax2 = plt.axes(projection='3d')
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            for (ox, oy, oz, size) in obstacleList:
                x = ox + size * np.outer(np.cos(u), np.sin(v))
                y = oy + size * np.outer(np.sin(u), np.sin(v))
                z = oz + size * np.outer(np.ones(np.size(u)), np.cos(v))
                ax2.plot_surface(x, y, z, color='r')
            ax.plot3D([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-k')
            plt.grid(True)
            plt.pause(0.001)    
            plt.show()


if __name__ == '__main__':
    main()
