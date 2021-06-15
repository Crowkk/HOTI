import kwant
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from scipy import spatial
from itertools import product
import copy

#crystal and random lattices
#kron product
#eigenvalues



sigma0 = np.identity(2)
sigma1= np.array([[0,1],[1,0]])
sigma2= np.array([[0,-1j],[1j,0]])
sigma3= np.array([[1,0],[0,-1]])
kron31 = np.kron(sigma3,sigma1)
kron02 = np.kron(sigma0,sigma2)
kron03 = np.kron(sigma0,sigma3)
kron11 = np.kron(sigma1,sigma1)

def lattice(basis,L,M=0,g=1,t1=1,t2=1,R=4,PBC =False):
    syst = kwant.Builder()
    
    lat = kwant.lattice.general([(0,L),(L,0)],basis,norbs=2) #I trick Kwant here I define a basis of N random sublattices
#and say one unit cell is a square LxL, then my entire system is contained in that one unit cell.

    def onsite(site):
        return np.kron(sigma0,sigma3)*(M+2*t2)
    

    def hopping(dist,theta):
        #here I'm not using site1, site1 because working directly with the positions is more convenient when using PBC
        #using the same f(\phi) from the paper

        C = np.cos(theta)
        S = np.sin(theta)
        f1 = -1j*t1*(kron31*C+kron02*S)
        f2 = -t2*kron03
        f3 = g*kron11*np.exp(1j*2*theta)
        f_th = f1+f2+f3

        #print(round(dist,3),round(theta/np.pi,3), round(C,3), round(S,3))
        return 0.5*f_th*np.exp(-dist)

    for i in range(len(basis)): #only the sites in the basis "exist"
        syst[lat.sublattices[i](0,0)] = onsite #actually only in the (0,0) exists i.e., the first unit cell


    bond = bondage(basis,PBC,R,size = L) #calling the bonds fucntion that outputs which sites


    for i in range(len(bond)):
        for j in range(len(bond[i])):
            index, theta,d = bond[i][j]
            index = int(index)
            syst[lat.sublattices[i](0,0),lat.sublattices[index](0,0)] = hopping(d,theta) #hopping between two sites


    """able = True
    if able:
        count = list()
        for b in bond:
            count.append(len(b))"""
    return syst.finalized()#,count

def bondage(lat,PBC,r,size):

    """function to return whom is bonded to whom

        Parameters:
        lat: all the sites in the lattice
        PBC: whether or not it has PBC
        r: the cut-off distance for bonding
        size: the OBC sample size
    """
    def distance(vec_1,vec_2):
        vec_distance = vec_2 - vec_1
        dist_size = np.sqrt(np.dot(vec_distance,vec_distance))
        return vec_distance,dist_size #returns both the vector and its length

    def angle(vec_1,vec_2):
        vec_distance, dist_size = distance(vec_1,vec_2)
        x_axis = np.array([1,0])
        theta = np.arctan2(vec_distance[1], vec_distance[0])
        if theta < 0:
            theta += 2*np.pi    
        return theta,dist_size #returns the angle (-np.pi, np.pi] and the distance between the sites


    og_size = len(lat) #how many sites in OBC
    orientations = [0,-1,1] 
    out = copy.deepcopy(lat)
    if PBC:
        for off in list(product(orientations,orientations))[1:]: #skip (0,0) i.e., the OBC
            off = np.array(off)
            out = np.concatenate((out,lat+size*off)) #out is a list with all the PBC copies of lat
    
    tree = spatial.KDTree(out) #the KDTree is on out
    bonds = tree.query_ball_point(x = lat, r = r) #find points in lat that are within distance r of out
    info_bond = list() #adjusted bonds

    for i in range(len(bonds)):
        b = bonds[i]
        b.remove(i) #remove onsite
        partial_b = list()
        for item in b:
            new_index = item
            if item >= og_size: #not in OBC
                new_index = item - int(item/og_size)*og_size #readjusts from a PBC copy to the index in the OBC
            ang,dist = angle(lat[i],out[item]) #calculate the angle and the distance
            #each = [new_index, ang,dist,out[item],item] #a list with the OBC index, the angle and the distance
            each = [new_index, ang,dist] #a list with the OBC index, the angle and the distance
            partial_b.append(each)
        info_bond.append(partial_b)


    return info_bond

#bond = bondage(lat,True,4,20)



size = 20
number = 20
g = 1
a0 = size/number
#lat = np.array([[x*a0 + a0*0.5, y*a0 + a0*0.5] for x in range(number) for y in range(number)])
state = np.random.RandomState(1209)
lat = size*state.random_sample((number**2,2))
syst = lattice(lat,L=size,PBC=False,g=1)
plt.title("HOTI Eigenvalues - Random")
plt.xlabel("n")
plt.ylabel("E_n")
H = syst.hamiltonian_submatrix()
val,vec = sla.eigh(H)
plt.plot(val,'bo')
plt.show()


