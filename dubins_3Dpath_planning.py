import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import dubins_2Dpath_planning

def cosine_sine(theta):
    """ Returns the cosine and sine value of an angle. """
    return math.cos(theta), math.sin(theta)

def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / (np.linalg.norm(vector) + 1e-06) # Addition of small number to prevent division by zero

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def obtain_heading_fpa(unit_vect):
    psi = math.atan2(unit_vect[1], (unit_vect[0]+1e-06)) # Addition of small number to prevent division by zero
    gamma = math.atan(unit_vect[2]/(math.sqrt(unit_vect[0]**2 + unit_vect[1]**2)+1e-06)) # Addition of small number to prevent division by zero
    return psi, gamma

def R1(X_f_T, psi_f_T, gamma_f_T, Rmin, S0=-1):  
    Cpsi_f_T, Spsi_f_T = cosine_sine(psi_f_T)
    Cgamma_f_T, Sgamma_f_T = cosine_sine(gamma_f_T) 
     
    A = X_f_T[2]*Cgamma_f_T*Cpsi_f_T - X_f_T[0]*Sgamma_f_T
    B = (X_f_T[1]-S0*Rmin)*Sgamma_f_T - X_f_T[2]*Cgamma_f_T*Spsi_f_T
    D = -S0*Rmin*Sgamma_f_T 
    
    if (A**2 + B**2 - D**2) < 0:
        return None
    tmp = math.sqrt(A**2 + B**2 - D**2)
    
    mode = ["R"]
    psi_t0_T = math.atan2(A*D+B*tmp, B*D-A*tmp)
#    print("-----In R1-----",A,B,D,psi_t0_T)
    if psi_t0_T > 0:
        return None, mode
    return psi_t0_T, mode

def R2(X_f_T, psi_f_T, gamma_f_T, Rmin, S0=-1):  
    Cpsi_f_T, Spsi_f_T = cosine_sine(psi_f_T)
    Cgamma_f_T, Sgamma_f_T = cosine_sine(gamma_f_T)    
    
    A = X_f_T[2]*Cgamma_f_T*Cpsi_f_T - X_f_T[0]*Sgamma_f_T
    B = (X_f_T[1]-S0*Rmin)*Sgamma_f_T - X_f_T[2]*Cgamma_f_T*Spsi_f_T
    D = -S0*Rmin*Sgamma_f_T  
    
    if (A**2 + B**2 - D**2) < 0:
        return None
    tmp = math.sqrt(A**2 + B**2 - D**2)
    
    mode = ["R"]
    psi_t0_T = math.atan2(A*D-B*tmp, B*D+A*tmp)
#    print("-----In R2-----",A,B,D,psi_t0_T)    
    if psi_t0_T > 0:
        return None, mode
    return psi_t0_T, mode

def L1(X_f_T, psi_f_T, gamma_f_T, Rmin, S0=1):   
    Cpsi_f_T, Spsi_f_T = cosine_sine(psi_f_T)
    Cgamma_f_T, Sgamma_f_T = cosine_sine(gamma_f_T)     
    
    A = X_f_T[2]*Cgamma_f_T*Cpsi_f_T - X_f_T[0]*Sgamma_f_T
    B = (X_f_T[1]-S0*Rmin)*Sgamma_f_T - X_f_T[2]*Cgamma_f_T*Spsi_f_T
    D = -S0*Rmin*Sgamma_f_T    
    
    if (A**2 + B**2 - D**2) < 0:
        return None
    tmp = math.sqrt(A**2 + B**2 - D**2)
    
    mode = ["L"]
    psi_t0_T = math.atan2(A*D+B*tmp, B*D-A*tmp)
#    print("-----In L1-----",A,B,D,psi_t0_T)    
    if psi_t0_T < 0:
        return None, mode
    return psi_t0_T, mode

def L2(X_f_T, psi_f_T, gamma_f_T, Rmin, S0=1):   
    Cpsi_f_T, Spsi_f_T = cosine_sine(psi_f_T)
    Cgamma_f_T, Sgamma_f_T = cosine_sine(gamma_f_T)     
    
    A = X_f_T[2]*Cgamma_f_T*Cpsi_f_T - X_f_T[0]*Sgamma_f_T
    B = (X_f_T[1]-S0*Rmin)*Sgamma_f_T - X_f_T[2]*Cgamma_f_T*Spsi_f_T
    D = -S0*Rmin*Sgamma_f_T    
    
    if (A**2 + B**2 - D**2) < 0:
        return None
    tmp = math.sqrt(A**2 + B**2 - D**2)
    
    mode = ["L"]
    psi_t0_T = math.atan2(A*D-B*tmp, B*D+A*tmp)
#    print("-----In L2-----",A,B,D,psi_t0_T)    
    if psi_t0_T < 0:
        return None, mode
    return psi_t0_T, mode

def Tplane_maneuver(X_f_T, ev_f_T, psi_f_T, gamma_f_T, Rmin):
    planners = [R1, R2, L1, L2]
    bcost = float("inf")
    bmode = None
    bpsi_t0_T = None
#    print("-----In Tplane_maneuver----- \n")
    for planner in planners:
        psi_t0_T, mode = planner(X_f_T, psi_f_T, gamma_f_T, Rmin)
#        print("psi_t0_T = ",psi_t0_T,"\n","mode = ",mode,"\n")
        if psi_t0_T is not None:
            if mode[0] == "R":
                S0 = -1
            else:
                S0 = 1
            
            # Verification of the obtained psi_t0
            X_t0_T = np.array([S0*Rmin*math.sin(psi_t0_T), S0*Rmin*(1-math.cos(psi_t0_T)), 0.0])    
            Cpsi_t0_T, Spsi_t0_T = cosine_sine(psi_t0_T)
            X1_f_T = X_f_T + ev_f_T
            X1_t0_T = X_t0_T + np.array([Cpsi_t0_T, Spsi_t0_T, 0.0])
            
            a = X1_f_T[2]*(X_f_T[1]-X_t0_T[1]) - X_f_T[2]*(X1_f_T[1]-X_t0_T[1])
            b = X_f_T[2]*(X1_f_T[0]-X_t0_T[0]) - X1_f_T[2]*(X_f_T[0]-X_t0_T[0])
            c = (X_f_T[0]-X_t0_T[0])*(X1_f_T[1]-X_t0_T[1]) - (X1_f_T[0]-X_t0_T[0])*(X_f_T[1]-X_t0_T[1])
        
            # Normal vectors of T plane and P plane
            NT = np.array([0, 0, 1])
            NP = np.array([a, b, c])
        
            GAMMA = angle_between(NT, NP)
            CGAMMA, SGAMMA = cosine_sine(GAMMA)
            # Transformation matrix from T-plane to P-plane 
            TtoP = np.array([[Cpsi_t0_T, Spsi_t0_T, 0.0],
                             [-Spsi_t0_T*CGAMMA, Cpsi_t0_T*CGAMMA, SGAMMA],
                             [Spsi_t0_T*SGAMMA, -Cpsi_t0_T*SGAMMA, CGAMMA]])
            
            X_f_P = TtoP.dot(X_f_T - X_t0_T)
            # Z-coordinate of destination waypoint in P-plane must be zero
            if(X_f_P[2] <= 1e-03):
                cost = abs(psi_t0_T*Rmin)
                if bcost > cost:
                    bpsi_t0_T, bmode = psi_t0_T, mode
                    bcost = cost
#    print("-----Sending to generate course---- \n bpsi_t0_T = ",bpsi_t0_T,"\n","bmode = ",bmode,"\n")        
    px_T, py_T, pz_T, ppsi_T, pgamma_T = generate_course_Tplane(bpsi_t0_T, bmode, Rmin)
    return px_T, py_T, pz_T, ppsi_T, pgamma_T, bcost, bmode
    
def generate_course_Tplane(psi_t0_T, mode, Rmin):
    if mode[0] == "R":
        S0 = -1
    else:
        S0 = 1
    
    px_T, py_T, pz_T, ppsi_T, pgamma_T = [], [], [], [], []
    pathpoints = 21
#    print("-----IN generate course Tplane----- \n psi_t0_T = ",psi_t0_T)
    for psi in np.linspace(0, psi_t0_T, pathpoints):
        px_T.append(S0*Rmin*math.sin(psi))
        py_T.append(S0*Rmin*(1-math.cos(psi)))
        pz_T.append(0.0)
        ppsi_T.append(psi)
        pgamma_T.append(0.0)

    return px_T, py_T, pz_T, ppsi_T, pgamma_T

def transform_trajectory(px_current, py_current, pz_current, ref_origin, TransMat):
    px_target, py_target, pz_target = [], [], []
    for x, y, z in zip(px_current, py_current, pz_current):
        px_target.append(TransMat[0][0]*x + TransMat[0][1]*y + TransMat[0][2]*z + ref_origin[0])
        py_target.append(TransMat[1][0]*x + TransMat[1][1]*y + TransMat[1][2]*z + ref_origin[1])
        pz_target.append(TransMat[2][0]*x + TransMat[2][1]*y + TransMat[2][2]*z + ref_origin[2])
    return px_target, py_target, pz_target
    
def transform_crossingdir(psi_current, gamma_current, TransMat):
    psi_target, gamma_target = [], []
    for (psi,gamma) in zip(psi_current, gamma_current):
        Cpsi, Spsi = cosine_sine(psi)
        Cgamma, Sgamma = cosine_sine(gamma)        
        ev_current = np.array([Cgamma*Cpsi, Cgamma*Spsi, Sgamma])
        ev_target = TransMat.dot(ev_current)     

        psi_, gamma_= obtain_heading_fpa(ev_target)        
        psi_target.append(psi_)
        gamma_target.append(gamma_)
        
    return psi_target, gamma_target

def dubins_3Dpath_planning(start_x, start_y, start_z, start_psi, start_gamma,
                            end_x, end_y, end_z, end_psi, end_gamma, Rmin):
    #
    Cpsi_i_I, Spsi_i_I = cosine_sine(start_psi)    
    Cgamma_i_I, Sgamma_i_I = cosine_sine(start_gamma)

    Cpsi_f_I, Spsi_f_I = cosine_sine(end_psi)
    Cgamma_f_I, Sgamma_f_I = cosine_sine(end_gamma)

    X_i_I = np.array([start_x, start_y, start_z])
    X_f_I = np.array([end_x, end_y, end_z])

    # Transformation matrix from Intertial to T-plane 
    ItoT = np.array([[Cgamma_i_I*Cpsi_i_I, Cgamma_i_I*Spsi_i_I, Sgamma_i_I], 
                    [-Spsi_i_I, Cpsi_i_I, 0.0],
                    [-Sgamma_i_I*Cpsi_i_I, -Sgamma_i_I*Spsi_i_I, Cgamma_i_I]])
    
    ev_f_I = np.array([Cgamma_f_I*Cpsi_f_I, Cgamma_f_I*Spsi_f_I, Sgamma_f_I])
    ev_f_T = ItoT.dot(ev_f_I)
    X_f_T = ItoT.dot(X_f_I-X_i_I)
    psi_f_T, gamma_f_T = obtain_heading_fpa(ev_f_T)
    
    # Path planning in T-plane
    px_T, py_T, pz_T, ppsi_T, pgamma_T, clen_T, mode_T = Tplane_maneuver(X_f_T, ev_f_T, psi_f_T, gamma_f_T, Rmin)        
    
    # Angle between T-plane and P-plane
    X1_f_T = X_f_T + ev_f_T
    X_t0_T = np.array([px_T[-1], py_T[-1], pz_T[-1]])
    psi_t0_T = ppsi_T[-1]

    Cpsi_t0_T, Spsi_t0_T = cosine_sine(psi_t0_T)
    X1_t0_T = X_t0_T + np.array([Cpsi_t0_T, Spsi_t0_T, 0.0])
    
    a = X1_f_T[2]*(X_f_T[1]-X_t0_T[1]) - X_f_T[2]*(X1_f_T[1]-X_t0_T[1])
    b = X_f_T[2]*(X1_f_T[0]-X_t0_T[0]) - X1_f_T[2]*(X_f_T[0]-X_t0_T[0])
    c = (X_f_T[0]-X_t0_T[0])*(X1_f_T[1]-X_t0_T[1]) - (X1_f_T[0]-X_t0_T[0])*(X_f_T[1]-X_t0_T[1])
    
    # Normal vectors of T plane and P plane
    NT = np.array([0, 0, 1])
    NP = np.array([a, b, c])

    GAMMA = angle_between(NT, NP)
    CGAMMA, SGAMMA = cosine_sine(GAMMA)
    # Transformation matrix from T-plane to P-plane 
    TtoP = np.array([[Cpsi_t0_T, Spsi_t0_T, 0.0],
                     [-Spsi_t0_T*CGAMMA, Cpsi_t0_T*CGAMMA, SGAMMA],
                     [Spsi_t0_T*SGAMMA, -Cpsi_t0_T*SGAMMA, CGAMMA]])
    
    X_f_P = TtoP.dot(X_f_T - X_t0_T)
    Cpsi_f_T, Spsi_f_T = cosine_sine(psi_f_T)
    Cgamma_f_T, Sgamma_f_T = cosine_sine(gamma_f_T) 
    ev_f_P = TtoP.dot(ev_f_T)

#    print("-----In Dubins_3D_path_planning-----")    
#    print("ItoT = ",ItoT,"\n","X_f_T = ",X_f_T,"\n","ev_f_I = ",ev_f_I,"\n","ev_f_T = ",ev_f_T,"\n","psi_t0_T = ",psi_t0_T,"\n","X1_f_T = ",X1_f_T,"\n")
#    print("psi_f_T = ",psi_f_T,"\n","gamma_f_T = ",gamma_f_T,"\n","X_t0_T = ",X_t0_T,"\n","X1_t0_T = ",X1_t0_T,"\n","a = ",a,"\n","b = ",b,"\n","c = ",c,"\n")
#    print("GAMMA = ",GAMMA,"\n","TtoP = ",TtoP,"\n","X_f_P = ",X_f_P,"\n","ev_f_P = ",ev_f_P,"\n")
    
    # Path Planning in P-plane
    psi_f_P = math.atan2(ev_f_P[1], ev_f_P[0]+1e-06)
    px_P, py_P, ppsi_P, mode_P, clen_P = dubins_2Dpath_planning.dubins_path_planning(0.0, 0.0, 0.0,
                                                    X_f_P[0], X_f_P[1], psi_f_P, 1/Rmin)
    pz_P, pgamma_P = [0.0]*len(px_P), [0.0]*len(ppsi_P)
    
    # Transform the path coordinates from P-plane to T-plane
    px_Ttemp, py_Ttemp, pz_Ttemp = transform_trajectory(px_P, py_P, pz_P, X_t0_T, TtoP.T)
    ppsi_Ttemp, pgamma_Ttemp = transform_crossingdir(ppsi_P, pgamma_P, TtoP.T)
    px_T, py_T, pz_T = px_T+px_Ttemp, py_T+py_Ttemp, pz_T+pz_Ttemp
    ppsi_T, pgamma_T = ppsi_T+ppsi_Ttemp, pgamma_T+pgamma_Ttemp
    # Transform the path coordinates from T-plane to inertial plane
    px_I, py_I, pz_I = transform_trajectory(px_T, py_T, pz_T, X_i_I, ItoT.T)
    ppsi_I, pgamma_I = transform_crossingdir(ppsi_T, pgamma_T, ItoT.T)
    
    return px_I, py_I, pz_I, ppsi_I, pgamma_I, clen_T+clen_P, mode_T+mode_P
    
if __name__ == '__main__':
    print("3D-Dubins Planner Start!!")    
    """
    3D-Dubins path plannner

    input:
        start_x : x position of start point [m]
        start_y : y position of start point [m]
        start_z : z position of start point [m]
        start_psi : heading angle of start point [rad]
        start_gamma : flight path angle of start point [rad]
        
        end_x : x position of end point [m]
        end_y : y position of end point [m]
        end_z : z position of end point [m]
        end_psi : heading angle of end point [rad]
        end_gamma : flight path angle of end point [rad]
        
        Rmin : minimum turning radius [m]

    output:
        px : x coordinates of path
        py : y coordinates of path
        pz : z coordinates of path
        ppsi : heading angle of path points
        pgamma: flight path angle of path points
        mode : type of curve

    """
    
    start_x = 0
    start_y = 2
    start_z = -1
    start_psi = np.deg2rad(10.0)
    start_gamma = np.deg2rad(-50.0) 
    
    end_x = 4
    end_y = 1
    end_z = -2
    end_psi = np.deg2rad(-10.0)
    end_gamma = np.deg2rad(-50.0) 
    
#    start_x = 0
#    start_y = 2
#    start_z = -1
#    start_psi = np.deg2rad(10.0)
#    start_gamma = np.deg2rad(-50.0) 
#    
#    end_x = 4
#    end_y = 1
#    end_z = -2
#    end_psi = np.deg2rad(-10.0)
#    end_gamma = np.deg2rad(-20.0)
    
    Rmin = 1         
    flag = 1
    try:
        px, py, pz, ppsi, pgamma, clen, mode = dubins_3Dpath_planning(start_x, start_y, start_z, start_psi, start_gamma,
                                                      end_x, end_y, end_z, end_psi, end_gamma, Rmin)
        print("Trajectory Type ---->",mode," and Length ----->",clen)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(px, py, pz, 'black')
    except:
        print("NOT POSSIBLE")
    





