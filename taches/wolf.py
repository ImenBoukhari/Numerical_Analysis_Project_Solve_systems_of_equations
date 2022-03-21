import numpy

compteurappels = 0


def f(x):
    global compteurappels
    compteurappels += 1
    denom = 1 + x[0] ** 2 + 3 * x[1] ** 2
    return -1 / denom


def gradf(x):
    global compteurappels
    compteurappels += 1
    denom = 1 + x[0] ** 2 + 3 * x[1] ** 2
    v = numpy.array([x[0], 3 * x[1]])
    return 2 / denom ** 2 * v


c1 = 0.1
c2 = 0.7


def pasWolfe(x, d, ainit, h0, gfx, Nmax=30):
    hp0 = numpy.vdot(gfx, d)
    c1hp0 = c1 * hp0
    c2hp0 = c2 * hp0
    amin = ainit
    hamin = f(x + amin * d)
    for i in range(Nmax):
        amax = amin
        hamax = hamin
        if hamin <= h0 + amin * c1hp0:
            break
        amin = amin / 2
        hamin = f(x + amin * d)
    for i in range(Nmax):
        gfxmin = gradf(x + amin * d)
        if numpy.vdot(gfxmin, d) >= c2hp0:
            return amin, hamin, gfxmin
        if hamax > h0 + amax * c1hp0:
            break
        amin = amax
        hamin = hamax
        amax = 2 * amin
        hamax = f(x + amax * d)
    for i in range(Nmax):
        a = (amin + amax) / 2
        ha = f(x + a * d)
        gfxa = gradf(x + a * d)
        if ha <= h0 + amax * c1hp0:
            if numpy.vdot(gfxa, d) >= c2hp0:
                break
            else:
                amin = a
        else:
            amax = a
    return a, ha, gfxa


def descenteGradWolfe(x0, tol, ainit=1, Nmax=1000):
    global compteurappels
    compteurappels = 0
    listecompteur = []
    listenormes = []
    listef = []
    stecompteur = []
    x = x0
    gfx = gradf(x0)
    ngfx = numpy.linalg.norm(gfx)
    fx = f(x)
    listenormes.append(ngfx)
    listef.append(fx)
    stecompteur.append(compteurappels)
    a = ainit
    for i in range(Nmax):
        if ngfx <= tol:
            break
        d = -gfx
        a, fx, gfx = pasWolfe(x, d, a, fx, gfx)
        x = x + a * d
        ngfx = numpy.linalg.norm(gfx)
        listenormes.append(ngfx)
        listef.append(fx)
        listecompteur.append(compteurappels)
    return x, listecompteur, listenormes, listef


x0 = numpy.array([2, 2])
x, lc, ln, lf = descenteGradWolfe(x0, 1e-5, 0.2)
print("pas entre =", x)
for i in range(0, len(lc)):
    print(" compteur = ", lc[i], " et normes =", ln[i], lf[i])