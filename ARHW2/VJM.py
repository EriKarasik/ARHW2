from numpy import transpose, identity as I, zeros, array, pi,hstack as h,vstack as v,cos,sin, arange, arctan2
from numpy.linalg import pinv, multi_dot as dot, inv
import matplotlib.pyplot as plt

Ka,E,G,L,d = 10000,7.0000e+10,2.5500e+10,0.75,0.05  # actuator stiffness
A,Iy,Iz,Ip = pi*(d**2)/4,pi*(d**4)/64,pi*(d**4)/64,2*pi*(d**4)/64

def Rx(q): return [[1,0,0,0],[0,cos(q),-sin(q),0],[0,sin(q),cos(q),0],[0,0,0,1]]
def dRx(q):return [[0,0,0,0],[0,-sin(q),-cos(q),0],[0,cos(q),-sin(q),0],[0,0,0,0]]
def Ry(q): return [[cos(q),0,sin(q),0],[0,1,0,0],[-sin(q),0,cos(q),0],[0,0,0,1]]
def dRy(q):return [[-sin(q),0,cos(q),0],[0,0,0,0],[-cos(q),0,-sin(q),0],[0,0,0,0]]
def Rz(q): return [[cos(q),-sin(q),0,0],[sin(q),cos(q),0,0],[0,0,1,0],[0,0,0,1]]
def dRz(q):return [[-sin(q),-cos(q),0,0],[cos(q),-sin(q),0,0],[0,0,0,0],[0,0,0,0]]
def Tx(q): return [[1,0,0,q],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
def dTx(): return [[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
def Ty(q): return [[1,0,0,0],[0,1,0,q],[0,0,1,0],[0,0,0,1]]
def dTy(): return [[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]
def Tz(q): return [[1,0,0,0],[0,1,0,0],[0,0,1,q],[0,0,0,1]]
def dTz(): return [[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]

th = [zeros(13), zeros(13), zeros(13)]
Tb = [dot([Ty(1),Ry(pi/2),Rz(pi)]), dot([Tz(1),Rx(-pi/2)]),I(4)]
Tt = [transpose(dot([Ry(pi/2),Rz(pi)])), transpose(Rx(-pi/2)), I(4)]

k11 = [[E*A/L,0,0,0,0,0],[0,12*E*Iz/L**3,0,0,0,6*E*Iz/L**2],[0,0,12*E*Iy/L**3,0,-6*E*Iy/L**2,0],[0,0,0,G*Ip/L,0,0],
       [0,0,-6*E*Iy/L**2,0,4*E*Iy/L,0],[0,6*E*Iz/L**2,0,0,0,4*E*Iz/L]]
k12 = [[-E*A/L,0,0,0,0,0],[0,-12*E*Iz/L**3,0,0,0,-6*E*Iz/L**2],[0,0,-12*E*Iy/L**3,0,6*E*Iy/L**2,0],[0,0,0,-G*Ip/L,0,0],
       [0,0,-6*E*Iy/L**2,0,2*E*Iy/L,0],[0,6*E*Iz/L**2,0,0,0,2*E*Iz/L]]
k22 = [[E*A/L,0,0,0,0,0],[0,12*E*Iz/L**3,0,0,0,-6*E*Iz/L**2],[0,0,12*E*Iy/L**3,0,6*E*Iy/L**2,0],[0,0,0,G*Ip/L,0,0],
     [0,0,6*E*Iy/L**2,0,4*E*Iy/L,0],[0,-6*E*Iz/L**2,0,0,0,4*E*Iz/L]]
k21 = transpose(k12)
#K = v([h([k11,k12]),h([transpose(k12), k22])])
Kth = v([h([Ka,zeros(12)]),h([zeros((6,1)),k22,zeros((6,6))]),h([zeros((6,1)),zeros((6,6)),k22])])

def JacobianPassiveLeg(tFk, tb, Tt, qa, qp, th):
    tFk[0:3, 3] = 0
    dt = dot([tb,dot([Tz(qa),Tz(th[0]),dRz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]),Tt,transpose(tFk)])
    J1 = v([dt[0,3],dt[1,3],dt[2,3],dt[2,1],dt[0,2],dt[1,0]])
    dt = dot([tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        dRz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]),Tt,transpose(tFk)])
    J2 = v([dt[0,3],dt[1,3],dt[2,3],dt[2,1],dt[0,2],dt[1,0]])
    dt = dot([tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),dRz(qp[2])]),Tt,transpose(tFk)])
    J3 = v([dt[0,3],dt[1,3],dt[2,3],dt[2,1],dt[0,2],dt[1,0]])
    return h([J1,J2,J3])

def JacobianthLeg(TFk,Tb,Tt,qa,qp,th):
    TFk[0:3, 3] = 0
    dt = dot([Tb,dot([Tz(qa),dTz(),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]), Tz(th[9]), Rx(th[10]), Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J1 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),dTx(),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]), Tz(th[9]), Rx(th[10]), Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J2 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),dTy(),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J3 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),dTz(),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J4 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),dRx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]), Tz(th[9]), Rx(th[10]), Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J5 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),dRy(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]), Rx(th[10]), Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J6 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),dRz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]), Rx(th[10]), Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J7 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),dTx(),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J8 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),dTy(),Tz(th[9]),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J9 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),dTz(),Rx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J10 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),dRx(th[10]),Ry(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J11 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),dRy(th[11]),Rz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J12 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    dt = dot([Tb,dot([Tz(qa),Tz(th[0]),Rz(qp[0]),Tx(0.75),Tx(th[1]),Ty(th[2]),Tz(th[3]),Rx(th[4]),Ry(th[5]),Rz(th[6]),
        Rz(qp[1]),Tx(0.75),Tx(th[7]),Ty(th[8]),Tz(th[9]),Rx(th[10]),Ry(th[11]),dRz(th[12]),Rz(qp[2])]), Tt, transpose(TFk)])
    J13 = v([dt[0, 3], dt[1, 3], dt[2, 3], dt[2, 1], dt[0, 2], dt[1, 0]])
    return h([J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13])

xScatter, yScatter,zScatter,dScatter = [],[],[],[]

F = array([[1], [0], [0], [0], [0], [0]])
for z in arange(0, 1, 0.1):
    for x in arange(0, 1, 0.1):
        for y in arange(0, 1, 0.1):
            qa = [x, y, z]
            Kc = 0
            for leg in range(len(Tb)):
                ploc = transpose(Tb[leg][0:3, 0:3]).dot(array([x, y, z]) - Tb[leg][0:3, 3])
                cos_q2 = (ploc[0] ** 2 + ploc[1] ** 2 - 0.75 ** 2 - 0.75 ** 2) / (2 * 0.75 * 0.75)
                sin_q2 = (1 - cos_q2 ** 2) ** 0.5
                q2 = arctan2(sin_q2, cos_q2)
                q1 = arctan2(ploc[1],ploc[0])-arctan2(0.75 * sin(q2),0.75+0.75*cos(q2))
                q3 = -(q1 + q2)
                qleg = [q1,q2,q3]
                TLeg = dot([Tb[leg],dot([Tz(qa[leg]),Tz(th[leg][0]),Rz(qleg[0]),Tx(0.75),Tx(th[leg][1]),Ty(th[leg][2]),Tz(th[leg][3]),
                                   Rx(th[leg][4]),Ry(th[leg][5]),Rz(th[leg][6]),Rz(qleg[1]),Tx(0.75),Tx(th[leg][7]),Ty(th[leg][8]),
                                   Tz(th[leg][9]),Rx(th[leg][10]),Ry(th[leg][11]),Rz(th[leg][12]),Rz(qleg[2])]), Tt[leg]])
                Jth = JacobianthLeg(TLeg,Tb[leg],Tt[leg],qa[leg],qleg,th[leg])
                Jq = JacobianPassiveLeg(TLeg,Tb[leg],Tt[leg],qa[leg],qleg,th[leg])
                Kc0 = inv(dot([Jth, inv(Kth), transpose(Jth)]))
                Kc += Kc0 - dot([Kc0,Jq,pinv(dot([transpose(Jq),Kc0,Jq])),transpose(Jq),Kc0])

            dt = inv(Kc).dot(F)
            deflection = (dt[0] ** 2 + dt[1] ** 2 + dt[2] ** 2)**.5
            xScatter.append(x)
            yScatter.append(y)
            zScatter.append(z)
            dScatter.append(deflection[0])

cmap = plt.cm.get_cmap('RdGy_r', 12)

#for i in range(len(dScatter)): dScatter[i] = log(dScatter[i])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
cmap = ax.scatter3D(xScatter, yScatter, zScatter, c=dScatter, cmap=cmap, s=60)
plt.colorbar(cmap)
plt.show()