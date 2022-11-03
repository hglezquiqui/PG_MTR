import numpy as np
from numpy.linalg import inv


def objective_function(A, x, z, b, lamda):
    # assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0) and \
    # np.size(x,1)== np.size(b,1) == 1 and np.isscalar(lamda))
    return f(A, z, x, b) + lamda*np.sum(np.sum(np.abs(x))) + lamda*np.sum(np.sum(np.abs(z)))


def f(A, z, x, b):
    #assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0))
    Ax_b = A.dot(x).dot(z) - b
    return 0.5*(np.trace(Ax_b.T.dot(Ax_b)))


def gradient_function(A, x, z, b):
    #assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0))
    return A.T.dot(A.dot(x).dot(z) - b).dot(z.T)


def ini_xk(A, z, b):
    x = inv(A.T.dot(A)).dot(A.T).dot(b).dot(z.T).dot(inv(z))
    return x


def gradient_function_z(A, x, z, b):
    #assert(np.size(x,0)==np.size(A,1) and np.size(A,0) == np.size(b,0))
    return x.T.dot(A.T).dot(A.dot(x).dot(z) - (b))


def model_function(x, xk, z, A, b, GammaK):
    # assert(np.size(xk,0) == np.size(x,0) == np.size(A,1) \
    # and np.size(A,0) == np.size(b,0) and np.isscalar(GammaK))
    innerProd = gradient_function(A, xk, z, b).T.dot(x - xk)
    xDiff = x - xk
    return f(A, z, xk, b) + np.trace(innerProd) + (1.0/(2.0*GammaK))*np.trace(xDiff.T.dot(xDiff))


def model_function_z(z, zk, x, A, b, GammaK):
    # assert(np.size(xk,0) == np.size(x,0) == np.size(A,1) \
    # and np.size(A,0) == np.size(b,0) and np.isscalar(GammaK))
    innerProd = gradient_function_z(A, x, z, b).T.dot(z - zk)
    xDiff = z - zk
    return f(A, zk, x, b) + np.trace(innerProd) + (1.0/(2.0*GammaK))*np.trace(xDiff.T.dot(xDiff))


def proximal_Norm(y, lamda):
    # assert(np.size(y,1)==1)
    return np.sign(y)*np.maximum(np.zeros(np.shape(y)), np.abs(y)-lamda)


def lasso_pgfit(A, b):
    # Define parameters. Size of A is n x p
    p = np.size(A, 1)
    n = np.size(A, 0)

    kMax = 500   # Number of iteration
    beta = 0.75  # decreasing factor for line search
    q = np.size(b, 1)
    r = min(p, q)
    # Generate the sparse vector xStar
    # and Randomly set 20 elements
    """
  xStar = np.zeros((p,q))
  xStar[np.floor(p*np.random.rand(20,q)).astype(np.int)]=1
  xStar = xStar*np.random.normal(0,10,(p,q))

  # Generate A and b. b = Ax + error
  A = np.random.randn(n,p)
  b = A.dot(xStar) + np.random.randn(n,q)
  """
    lamda = np.sqrt(2*n*np.log(p)).tolist()
    #lamda = 1.

    # For Proximal Gradient Descent
    zk = np.identity(q) + 0.01*np.random.rand(r, q)
    # xk = np.random.rand(p,r) # Initialize with random
    # xk = np.zeros((p,q))      # Initialize with zero
    xk = ini_xk(A, zk, b)
    # For Accelerated Proximal Gradient Descent
    xk_acc = xk.copy()
    yk_acc = xk_acc
    tk_acc = 1
    zk_acc = zk.copy()
    zyk_acc = zk_acc
    ztk_acc = 1
    stopUpdate_acc = False
    Dobj_acc = 0
    cri = []
    fobj = []

    for k in range(kMax):

        # ------------------------- Proximal GD -----------------------------
        Gammak = 0.01
        #Gammak = 1/np.linalg.norm(A.T.dot(A))

        # Line search
        while True:
            #print ('trying stepsize = ', "{0:0.2e}".format(Gammak))
            # Gradient Descent (GD) Step
            x_kplus1 = xk - Gammak*gradient_function(A, xk, zk, b)

            # print(m(x_kplus1,xk,A,b,Gammak))
            if f(A, zk, x_kplus1, b) <= model_function(x_kplus1, xk, zk, A, b, Gammak):
                # print ' success'
                break
            else:
                # print ' Fail ',
                Gammak = beta*Gammak
        # Proximal Operation (Shrinkage)
        x_kplus1 = proximal_Norm(x_kplus1, Gammak*lamda)

        # Change in the value of objective funtion for this iteration
        Dobj = np.linalg.norm(
            objective_function(A, x_kplus1, zk, b, lamda) - objective_function(A, xk, zk, b, lamda))
        # print 'k:',k, ' obj = ', obj(A,x_kplus1,b,lamda), 'Change = ',Dobj

        # Update xk
        xk = x_kplus1.copy()
        # --------------------------------------------------------------------
        # ===================zUpdate GD Proximal===============================
        Gammak = 0.01
        #Gammak = 1/np.linalg.norm(A.T.dot(A))

        # Line search
        while True:
            #print ('trying stepsize = ', "{0:0.2e}".format(Gammak))
            # Gradient Descent (GD) Step
            z_kplus1 = zk - Gammak*gradient_function_z(A, xk, zk, b)

            # print(m(x_kplus1,xk,A,b,Gammak))
            if f(A, z_kplus1, xk, b) <= model_function_z(z_kplus1, zk, xk, A, b, Gammak):
                # print ' success'
                break
            else:
                # print ' Fail ',
                Gammak = beta*Gammak
        # Proximal Operation (Shrinkage)
        z_kplus1 = proximal_Norm(z_kplus1, Gammak*lamda)

        # Change in the value of objective funtion for this iteration
        Dobjz = np.linalg.norm(
            objective_function(A, xk, z_kplus1, b, lamda) - objective_function(A, xk, zk, b, lamda))
        # print 'k:',k, ' obj = ', obj(A,x_kplus1,b,lamda), 'Change = ',Dobj

        # Update xk
        zk = z_kplus1.copy()
        # --------------------------------------------------------------------

        cri.append(objective_function(A, xk_acc, zk_acc, b, lamda))
        fobj.append(objective_function(A, xk, zk, b, lamda))
        # Terminating Condition
        if (min(Dobj, Dobjz) < 1):
            break
    return xk, zk, cri, fobj
