import numpy as np                   #-------------- Mohammad Omar 1212429 --------------#
import math                          #-------------- Miassar Johar 1210519 --------------#
import random                        #-------------- Taleen Bayatnrh 1211305 ------------#
import matplotlib.pyplot as plt      #-------------- Hamza Barhosh 1210920 --------------#
                                   
# ------------------------- Maze Generation ------------------------- #
N = 6  # build a maze thats 6 cells by 6 cells.
cell = 0.1  # each cell is 0.1 meters 10 cm on a side.
random.seed(2)  # setting a seed so we get the same maze every time ehich helps with debugging.

grid = np.full((N, N), 1|2|4|8, dtype=int)  # create a grid and every cell start with all four walls intact so we use bits to represent walls: 1 (North), 2 (East), 4 (South), 8 (West) and this (1|2|4|8) does a bitwise OR to combine all these bits into one value (15).
visited = np.zeros_like(grid, dtype=bool) # array that keep track the cells that visited so we dont process them twice.
dirs = [(0, -1, 1, 4), (1, 0, 2, 8), (0, 1, 4, 1), (-1, 0, 8, 2)] #define the 4 possible directions: up, right, down, and left and for each direction, we include: dx, dy for how to move in the grid, w: the bit mask for the wall to remove from the current cell, opp: the bit mask for the wall to remove from the neighbor the opposite wall
def carve(x, y): # carve mean randomly knocks down walls between cells until all cells are connected in the maze using depth-first search (DFS),x, y - the current cell's position in the grid.
    visited[y, x] = True  # mark the current cell as visited,
    random.shuffle(dirs) # make the directions random
    for dx, dy, w, opp in dirs: # try each dir from the current cell.
        nx, ny = x + dx, y + dy # compute the neighbor's grid index
        if 0 <= nx < N and 0 <= ny < N and not visited[ny, nx]:   #check if the neighbor is within bounds and doesnt visited yet.
            grid[y, x] &= ~w  # use bitwise AND with ~w to remove wall 'w' from the current cell.
            grid[ny, nx] &= ~opp #remove the opposite wall from the neighboring cell using the same way
            carve(nx, ny) # recursively carve from the neighbor cell.

carve(0, 0) # start maze generation from the upper-left cell (0,0)

walls = [] # create list to put the wall segments in it for plot.
for y in range(N):#loop on each cell to extract the remaining walls
    for x in range(N):
        cx, cy = x*cell, y*cell # compute the cells (x,y) pos in meters depend on its grid indi.
        w = grid[y, x]    # get the wall bitmask for the current cell
         #check all walls one by one using bitwise AND and add the corresponding segment if there is a segment
        if w & 1: # if bit 0 "value 1" is set,include the top wall segment
            walls.append((cx, cy, cx+cell, cy))
        if w & 2:   # if bit 1"value 2" is set,include the right wall segment
            walls.append((cx+cell, cy, cx+cell, cy+cell))
        if w & 4:  #if bit 2 "value 4" is set,include the bottom wall segment.
            walls.append((cx, cy+cell, cx+cell, cy+cell))
        if w & 8: # if bit 3 "value 8"is set include the left wall segment.
            walls.append((cx, cy, cx, cy+cell))

for i in range(N):# add the outer boundary walls of the maze "المحيط" ,then loop over each cell inde along the boundary to ensure the maze is enclosed.
    walls.append((i*cell, 0, (i+1)*cell, 0)) #the top boundary wall.
    walls.append((i*cell, N*cell, (i+1)*cell, N*cell)) #the bottom boundary wall.
    walls.append((0, i*cell, 0, (i+1)*cell))#the left boundary wall.
    walls.append((N*cell, i*cell, N*cell, (i+1)*cell)) #the right boundary wall.

# ------------------------- Geometry Utils ------------------------- #
def seg_intersect(p, q, a, b): #determine if two line segments (p,q) and (a,b) intersect using the concept of counterclockwise (ccw) orientation to decide if the segments intersect.
    (px, py), (qx, qy) = p, q  # p,q: tuples that represent the endpoint of the first segment.
    (ax, ay), (bx, by) = a, b  # p,q: tuples that represent the endpoint of the second segment.
    def ccw(A, B, C): #define an inner helper function to test counterclockwise order by Check if three points A, B, C are listed in counterclockwise order.
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0]) # this is done by comparing the slopes of cross-product so if the result is True,then the three points make a counterclockwise turn
    return ccw(p, a, b) != ccw(q, a, b) and ccw(p, q, a) != ccw(p, q, b)  # the two segments (p,q) and (a,b) intersect if:1)p, a, b are not all in the same orientation as q, a, b, AND, 2)p, q, a are not all in the same orientation as p, q, b.

def raycast_distance(x, y, theta, max_dist=1.0):#function  to Cast a ray from the starting point (x, y) in the direction of theta and compute the distance to the closest intersection with a wall, args : x float , x coord of the start point , y float ,y coord of start poinr,theta float , the angle in radian of the ray direc , max dist float, max dist for the ray and then return min d float , min dis from x,y to any intersection wall
    min_d = max_dist #init shortest dis to the max ray length
    sx, sy = math.cos(theta), math.sin(theta) # compute the unit direction vector sx, sy for the ray
    for x1, y1, x2, y2 in walls: # loop over all wall segment that  stored in the wall list
        dx, dy = x2 - x1, y2 - y1 # compute the directional vector dx, dy of the wall segment.
        denom = dx*sy - dy*sx # calc the denom "determinant " which is part o intersection  calcs using the cross product between the rays dire and the walls dire
        if abs(denom) < 1e-9:  # if denom is almmost zero, so that mean the ray and the wall are parallel or very close to it,so there is no intersection to consider for this wall.
            continue
        t = ((x1 - x)*sy - (y1 - y)*sx) / denom #compute the parameter t which it is indicate where along the wall segment the intersection is happen
        u = ((x1 - x)*dy - (y1 - y)*dx) / denom  # compute the parameter u which indicates how far along the ray the intersection happen
        if 0 <= t <= 1 and u >= 0:   # check if the intersection point is on the wall segment (0 <= t <= 1) and in the forward direction along the ray (u >= 0)
            min_d = min(min_d, u)  # update min_d if this intersection is closer than any previous intersections.
    return min_d #return the shortest distance "best dist"
# ------------------------- Robot Parameters ------------------------- #
# physical parameters of the micromouse robot
r = 0.02 # wheel radius, m
d = 0.1 # dist between the two wheels, m
m = 0.2  # mass of the robot ,kg
I = 0.0005 # moment of inertia about the vertical z axis, kg·m^2
km = 0.1   # motor constant, N·m/V ,relates applied voltage to the motors torque
#friction coefficients representing 
b = 0.01 # linear friction coefficient N·s/m
b_theta = 0.001 #angular or rotational friction coefficient N·m·s/rad
dt = 0.05 # simulation time step for numerical integration it is also used in the discrete simulation update
Q = np.diag([0.1]*5) * dt # process the noise covariance matrix for the EKF multiplied by dt to scale the noise over the interval,the diagonal covariance is assumed for the five-state elements [x,y,theta,v,omega]
#measure the noice covar matrix for ekf
R = np.diag([0.05**2, 0.05**2, 0.02**2]) #the encoder measure have variance=(0.05)^2 and the gyroscope measure variance=(0.02)^2 ,the matrix is diagonal corresponding to the three measure channels
# command parameters for the controller
v_cmd_fwd = 0.12  # the needed forward linear velocity m/s when forward mode is on 
w_cmd_turn = math.pi / 2 #the needed angular velocity rad/s when turning mode is on turn 90 degrees per second
#sensor and controll 
front_thresh = 0.03  # if sth gets closer than 0.03 meters in front of the mouse , well consider it too close and decide to take evasive action.
sensor_noise = 0.005 #add a little bit of random noise about 0.005 std deviation to our raycast sensor readings to mimic real-world imperfections.
#simulation run time 
max_steps = int(1000 / dt)  # calculate the total number of steps by dividing 1000 by dt so this is the number of update steps we expect to run.

# ------------------------- Simulation State ------------------------- #
x_true = np.array([0.05, 0.05, 0.0, 0.0, 0.0]) # this array store the real true state of our robot,We start the robot at position (0.05, 0.05) with an orientation of 0 radians facing right and no initial motion.
x_est  = x_true.copy() #This is the robot estimated state ,We init it to match the true state since we assume a perfect start.
P = np.eye(5) * 0.01 #p is the estimation error covariance matrix for the EKF its measure of  uncertainty about the state estimate, starting off small 0.01 for each state 
mode = "forward" #mode is the variable that keeps track of what the robot is currently doing , forwrd mode mean the robot will try move straight 
turn_dir = 1 #mean that the direction to turn when needed is define by one ( 1 for turning one way and -1 for the opposite).
theta_start_turn = 0.0  #when a turn begins, we record the robots initial heading which help to determine how far it has turn 
target_xy = np.array([0.55, 0.57]) #the target position the robot is trying to reach
traj_true = []#a list that keep track of the robots actual path over time.
traj_est = [] #also this list will store the estimated trajectory as computed by our EKF.
recent_modes = [] # used to know the recent control modes which might help to debug or analyze the robot decision 

# ------------------------- Main Loop ------------------------- #
for step in range(max_steps):  
    #there are three directions for the robot to see through virtual distance sensors

    #look straight ahead
    front  = raycast_distance(*x_true[:2], x_true[2]) + np.random.normal(0, sensor_noise)
    #look to the left
    left   = raycast_distance(x_true[0], x_true[1], x_true[2] + math.pi/2) + np.random.normal(0, sensor_noise)
    #look to the right
    right  = raycast_distance(x_true[0], x_true[1], x_true[2] - math.pi/2) + np.random.normal(0, sensor_noise)


    # Escape if boxed in

    #if the robot detects very little space in all direcions it will switch to turn mode and rotate left to escape
    if front < 0.03 and left < 0.03 and right < 0.03:
        turn_dir = 1  # always turn left when totally boxed
        mode = "turn" #switch to turning behavior
        theta_start_turn = x_est[2]  #save the heading before the turn started

    #check if an obstacle is ahead

    #if we are currently trying to go foraward, but the front sensor sees a wall too close we decide to turn instead thats to avoid collision
    if mode == "forward" and front < front_thresh:
        #check which side has more room 
        #if right side is more open turn right, otherwise turn left
        turn_dir = 1 if right > left else -1 
        mode = "turn"  #switch to turn mode
        theta_start_turn = x_est[2]  #store our current heading so we can check later how far we have turned
    
    #check if turning is complete
     
    #while in turn mode keep turning until about 90 degree is completed
    elif mode == "turn":
        #calculate how much we have turned since we started
        #this formula normalizes the angle difference to between (-pi , pi)
        dtheta = ((x_est[2] - theta_start_turn + math.pi) % (2*math.pi)) - math.pi
        #if we have turned roughly 90 degrees switch back to forward motion 
        if abs(dtheta) >= math.pi/2 - 0.05:
            mode = "forward"
    
    #based on weather we are in forward or turn mode set our desired speeds
    #if mode == forward --> go forward at desired speed mo turning 
    #else -->rotate in place dont move forward
    v_des, omega_des = (v_cmd_fwd, 0.0) if mode == "forward" else (0.0, turn_dir * w_cmd_turn)


    #compare the current velocity estimates to the desired ones
    #we will use proportional control to close the gap

    #linear velocity error, angular velocity error
    e_v = v_des - x_est[3]; e_w = omega_des - x_est[4]

    #use proportional control to compute desired accelerations
    #the constants act like gains — higher values mean quicker correction
    vdot_des = 3.0 * e_v; wdot_des = 6.0 * e_w

    #V_sum controls the total voltage needed to reach vdot_des (for both wheels combined)
    V_sum  = (m * r / km) * (vdot_des + (b / m) * x_est[3])

    #V_diff controls the difference in voltage needed to turn (left vs right wheel)
    V_diff = (2 * I * r / (d * km)) * (wdot_des + (b_theta / I) * x_est[4])

    #combine the total and differential voltages to get each wheel's command:
    #V_left  = (V_sum - V_diff) / 2
    #V_right = (V_sum + V_diff) / 2
    #clip the voltages to stay within safe limits [-5V, +5V]
    V_l = np.clip(0.5 * (V_sum - V_diff), -5, 5)
    V_r = np.clip(0.5 * (V_sum + V_diff), -5, 5)

                                                      #!Important Note!#        >>>>>>> without these varible the robot cant move since he did not know were he is and what the currant state 
                  # True dynamics with collision check
    theta, v, w = x_true[2], x_true[3], x_true[4]     # theta will gives us the dircition of the robot id facing and the v will till us who fast it moing in a "straight line"
    vdot = km/(m*r)*(V_l + V_r) - (b/m)*v             # and w for turning , the propes of these varible is at the begining of each sumulation step (for us its 1000/0.5) each step he need to calculate 
    wdot = d*km/(2*I*r)*(V_r - V_l) - (b_theta/I)*w   # each varible , to undurstand its currant physical condition so TO know who we will move so its will till us about if he will go in linear way or angular way velocity 
    dx = v * math.cos(theta) * dt                     # so we can know calcalute every step by using the dynamic eqation and by there parameter
    dy = v * math.sin(theta) * dt                     # Vdot For calculate who much accelaration reqered to moce "JUST IN STRAIGHT LINE" (it comes from the Newton Second law) tells us how much the robot forward speed is changing due to the combined effects of the motors pushing and friction resisting.
    new_pos = np.array([x_true[0]+dx, x_true[1]+dy])  # wdot FOR angular acceleration its captures how quickly the robot is turning due to motor imbalance and how much that turn is resisted by internal friction
    crossed = any(seg_intersect((x_true[0],x_true[1]), new_pos, (x1,y1),(x2,y2)) for x1,y1,x2,y2 in walls)
                                                      # For every step we need to plot were does the robot are in 2D maze so we used these tow componant dx and dy and every step we called the mat plot
    

                    # WALL Detect Function #
    if crossed:                     
        left_live  = raycast_distance(x_true[0], x_true[1], x_true[2] + math.pi/2)
        right_live = raycast_distance(x_true[0], x_true[1], x_true[2] - math.pi/2)
        turn_dir = 1 if right_live > left_live else -1# 90 Degree Turn for left and right statmant 1+ for left and -1 for the right "dicreipe below"
        x_true[3] = 0                                 # Finaly we are in the maze Detect , the Function will Deceted if the robot is crossed or not 
        mode = "turn"                                 # after we detect the next step we need to move , we need to cheack if this step will crossed the wall of the maze
        theta_start_turn = x_est[2]                   # so of crossed is true , then we will enter the crossed if statment 
    else:                                             # and for this we will call our raycast sinsor , that will check we will go right or lefft ??
        x_true[0:2] = new_pos                         # this depened on the who realy far from any thhink on the left and right or nearest wall on the left and the right
        x_true[3] = v + vdot * dt                     # so the raycast will chose the best open side , and to avoid moving while it turn we set the velocity wile it move 0
                                                      # before turn , store the current estimated orientation angle what we discipe it before
    x_true[2] += w * dt
    x_true[2] = ((x_true[2] + math.pi) % (2*math.pi)) - math.pi
    x_true[4] = w + wdot * dt                         # so if there is no walls , the statment goes to else , that will update the new currrant positon and update its linear velocity Vdot and orientation angle 

                # Sensor measurements part
    v_L = x_true[3] - (d/2)*x_true[4]                 # sensor readings specifically the encoders on the wheels and the gyroscope
    v_R = x_true[3] + (d/2)*x_true[4]                 # v_l and v_r are the left and right wheel velocities derived from the robots linear and angular speed
    z = np.array([                                    # they are converted to angular speeds (what encoders would read) and To make it realistic, gaussian noise is added
        v_L / r + np.random.normal(0, 0.05),          # so the he final vector z contains the noisy encoder and gyro readings used for EKF correction
        v_R / r + np.random.normal(0, 0.05),
        x_true[4] + np.random.normal(0, 0.02)
    ])


                                                      #<Emportant note>#

                     # EKF Part #    >>>>>            <"it tells us how each state variable affects the others when the robot moves">
    theta_e = x_est[2]                                # this block estimates how the robots state will change based on its current estimated state
    xdot_est = np.array([                             # theta is the estimated heading used to project velocity into x and y directions  
        x_est[3]*math.cos(theta_e),                   # the third value is angular velocity "how fast the robot is rotating"
        x_est[3]*math.sin(theta_e),                   # the last two values compute linear and angular acceleration using motor inputs and friction
        x_est[4],                                     # this full xdot_est vector is used to predict the next state in the EKF
        km/(m*r)*(V_l + V_r) - (b/m)*x_est[3],
        d*km/(2*I*r)*(V_r - V_l) - (b_theta/I)*x_est[4]
    ])
    x_pred = x_est + xdot_est * dt                   # x_pred updates the estimated state using xdot_est predicting where the robot will be next
    x_pred[2] = ((x_pred[2] + math.pi) % (2*math.pi)) - math.pi

    F = np.eye(5)                                    # the angle x_pred[2] is wrapped to the range -π, π to keep heading consistent F is the Jacobian matrix that linearizes the motion model for EKF prediction
    F[0,2] = -x_est[3]*math.sin(theta_e)*dt          # here is the point that each partial derivative in F reflects how changes in state variables affect motion ok for example let  F in 0 to 3 = cos() shows how forward velocity affects x position
    F[0,3] =  math.cos(theta_e)*dt                   # so since F is an matric so ots represent every think on like i said before let F[1,2] will be accounts for how turning changes y-position due to rotation
    F[1,2] =  x_est[3]*math.cos(theta_e)*dt          # F[1,3] and F[3,3] repectivly is captures how velocity affects movement in the y-direction and the other is eflects how linear velocity naturally decays over time due to friction
    F[1,3] =  math.sin(theta_e)*dt
    F[2,4] =  dt 
    F[3,3] =  1 - (b/m)*dt
    F[4,4] =  1 - (b_theta/I)*dt
    P = F @ P @ F.T + Q                   #>>>>>>   # updates the uncertainty in the state prediction using model noise Q this prepares the EKF to correct the predicted state using sensor measurements in the next step


    H = np.zeros((3,5))                             # this section updates the state estimate using noisy sensor measurement
    H[0,3], H[0,4] = 1/r, -d/(2*r)                  # H is the measurement Jacobian mapping state variables to expected sensor readings encoders and gyro the rows of H describe how velocity and angular velocity influence left and right wheel speeds and rotation
    H[1,3], H[1,4] = 1/r,  d/(2*r)                  # v_L_est and v_R_est compute the expected encoder readings from the current state estimate also h is the predicted measurement vector, containing expected sensor outputs
    H[2,4]        = 1                               # S is the innovation covariance showing how uncertain the predicted measurement is and K is the Kalman Gain it decides how much to trust the sensors vs the prediction 
    v_L_est = x_est[3] - (d/2)*x_est[4]             # then y=z-h computes the difference between actual and expected sensor readings and aslo x_est = x_pred + K @ y corrects the predicted state using the measurement error
    v_R_est = x_est[3] + (d/2)*x_est[4]             # and for P  updates the uncertainty in the estimate after the correction
    h = np.array([v_L_est/r, v_R_est/r, x_est[4]])  # important line code for Predicted sensor readings based on current state that you used in the previous code
    S = H @ P @ H.T + R                             # so here we compute the innovation covariance matrix S as we said 
    K = P @ H.T @ np.linalg.inv(S)                  # calculate the Kalman Gain (K), which determines how much we should trust the
                                                    # measurement versus our model prediction a higher gain means we trust the sensor more
    y = z - h                                       # compute the innovation vector y as we said in point five
    x_est = x_pred + K @ y                          # Update the state estimate x_est by correcting the predicted state x_pred using the Kalman Gain and the innovation this brings the estimate closer to the real state
    x_est[2] = ((x_est[2] + math.pi) % (2*math.pi)) - math.pi 
    P = (np.eye(5) - K @ H) @ P                     # # Wrap the robot’s orientation (theta) back to the range [-π, π] to keep angle math consistent this is important for systems that deal with circular quantities like rotation 

    traj_true.append(x_true.copy())                 # then Store true state for plotting and also Store estimated state for plotting , in our case we can use both of thim but we 
    traj_est.append(x_est.copy())                   # we stick in the true path , and also we can geerate both of them since we have the path saved


# ------------------------- Plot ------------------------- #
traj_true = np.array(traj_true)                    # convert the list of true robot states into a NumPy array for easier indexing and plotting
traj_est = np.array(traj_est)                      # convert the list of estimated states (from EKF) into a NumPy array for plotting

plt.figure(figsize=(6, 6))                         # create a new square plot with a 6x6 inch size for clear visibility

for (x1, y1, x2, y2) in walls:                     # loop through all wall segments defined as line coordinates
    plt.plot([x1, x2], [y1, y2], linewidth=2, color="black")  # draw each wall segment as a thick black line

plt.plot(traj_true[:,0], traj_true[:,1], label="True Trajectory")         # plot the actual path taken by the robot using the true x and y positions

plt.scatter(*target_xy, marker='*', s=120, color="gold", label="Target")  # mark the goal or target location as a large gold star

plt.axis('equal')                                   # ensure that one unit on the x-axis equals one unit on the y-axis (square aspect ratio)
plt.xlim(-0.02, N*cell + 0.02)                      # set the x-axis range slightly beyond the maze boundaries for better view
plt.ylim(-0.02, N*cell + 0.02)                      # set the y-axis range similarly for symmetry and margin

plt.title("Micromouse Maze – Continuous Motion with Corner Recovery")  # add a descriptive title to the plot

plt.xlabel("x (m)")                                 # label the x-axis as representing meters in the horizontal direction
plt.ylabel("y (m)")                                 # label the y-axis as representing meters in the vertical direction

plt.grid(True)                                      # display a background grid for better spatial reference
plt.legend()                                        # dhow the plot legend to distinguish between the trajectory and the target marker

plt.show()                                          # Finally display the plot window
