# Based on "Heuristic Solution of an Extended Double-Coverage Ambulance Location Problem for Austria" Karl F. Doerner et al.
from gekko import GEKKO
import numpy as np
from tabulate import tabulate
from math import ceil

model = GEKKO(remote=True)
model.options.SOLVER = 1  # select mixed integer solver APOPT

model.solver_options = ['minlp_maximum_iterations 500',
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 50',
                        # treat minlp as nlp
                        'minlp_as_nlp 0',
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 50',
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1',
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.05',
                        # covergence tolerance
                        'minlp_gap_tol 0.01']

dimW = 160            # decision variable  - Ambulance possible locations [j]
dimV = 250            # constraints number - Villages-Area to be covered  [i]
maxAmbulanceTot = 80  # Tot Number of Ambulances

max_Ambulance_per_location  = np.random.randint(low=2, high=4, size=(dimW))
omega                       = 0.8  # minimum fraction of total demand (sum lambda) that must be covered at least by an ambulance in radius r (small)

smallRadius = 3  # small radius to be covered
bigRadius   = 6  # big radius to be covered

timeMatrix    = np.random.randint(low=1, high=6, size=(dimV, dimW))   # time needed from each [j]  Ambulance possible Location to reach i village/district location [i]
lambdaMatrix  = np.random.randint(low=1, high=20, size=(dimV))        # number of people per district/village

gamma = (timeMatrix <= smallRadius) * 1   # whether the location[i] is within reach (small r) of ambulance in location [j]
delta = (timeMatrix <= bigRadius) * 1     # whether the location[i] is within reach (big R)   of ambulance in location [j]

z_decisionVariable = model.Array(model.Var, (dimW), integer=True)

# make decision Variable boolean
for kk in range(len(z_decisionVariable)):
    z_decisionVariable[kk].value = 1
    z_decisionVariable[kk].lower = 0
    z_decisionVariable[kk].upper = 1

# NB: Workaround 1. We need Intermediate Variable in Gekko otherwise the expression will exceed the character limit of 15000

# Constraint Eq 2
Eq2_constraints = delta @ z_decisionVariable
for kk in range(dimV):
    inter = model.Intermediate(Eq2_constraints[kk])
    model.Equation([inter >= 1])

# To build x_i and y_i I need intermediate variable, h[i] is the number of ambulance within r(small) per each Village [i]
h_i = [None] * dimV
for ii in range(dimV):
    somma = 0
    for jj in range(dimW):
        somma = somma + gamma[ii, jj] * z_decisionVariable[jj]
    h_i[ii] = model.Intermediate(somma)

# Boolean True if at least one ambulances per location [i] V within radius r (small)
x_i = [None] * dimV
for kk in range(len(x_i)):
    x_i[kk] = model.if3(h_i[kk] - 0.5, 0, 1)
    inter = model.Intermediate(x_i[kk])
    model.Equation(inter >= 1)

# Boolean True if at least two ambulances per location V within radius r (small)
y_i = [None] * dimV
for kk in range(len(y_i)):
    y_i[kk] = model.if3(h_i[kk] - 1.5, 0, 1)

# Constraints Eq 3 -
inter = model.Intermediate(model.sum(lambdaMatrix * np.asarray(x_i)))
inter2 = model.Intermediate(omega * model.sum(lambdaMatrix))
model.Equation([inter >= inter2])

# Constraints Eq 4
LHS = [None] * dimV
RHS = [None] * dimV
for ii in range(dimV):
    LHS[ii] = 0
    for jj in range(dimW):
        LHS[ii] = LHS[ii] + gamma[ii, jj] * z_decisionVariable[jj]

    RHS[ii] = x_i[ii] + y_i[ii]
    model.Equation([LHS[ii] >= RHS[ii]])

# Constraints Eq 5
for ii in range(dimV):
    model.Equation(y_i[ii] <= x_i[ii])

# Constraints Eq 6
model.Equation(model.sum(z_decisionVariable) == maxAmbulanceTot)

# Constraints Eq 7
for jj in range(dimW):
    model.Equation(z_decisionVariable[jj] <= max_Ambulance_per_location[jj])

# Objective function Eq 1)
for ii in range(dimV):
    model.Obj(-lambdaMatrix[ii] * y_i[ii])

model.solve(disp=True)

# reshaping and printing in table format
nCols = 10
nRows = ceil(dimW / 10)
data = -np.ones(shape=(nRows, nCols))

kk = 0
for ii in range(nRows):
    header = []
    for jj in range(nCols):
        if kk < dimW:
            data[ii, jj] = z_decisionVariable[kk][0]
        if kk < 9:
            header.append("var00" + str(kk + 1))
        if kk > 100:
            header.append("var" + str(kk + 1))
        if kk >= 9 and kk <= 100:
            header.append("var0" + str(kk + 1))
        kk += 1

    dataTemp = [data[ii, :]]
    table = tabulate(dataTemp, header, tablefmt="fancy_grid")
    print(table)
