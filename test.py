from gurobipy import Model, GRB

# 创建一个新的模型
m = Model("my_model")

# 添加变量，这里添加两个连续变量 x 和 y
x = m.addVar(name="x")
y = m.addVar(name="y")

# 设置目标函数，这里是一个简单的线性目标函数
m.setObjective(x + y, GRB.MAXIMIZE)

# 添加约束条件
m.addConstr(2*x + y <= 10, "c0")
m.addConstr(x + 2*y <= 8, "c1")
m.addConstr(x >= 0, "c2")
m.addConstr(y >= 0, "c3")

# 优化模型
m.optimize()
print(f"Optimal solution found: x = {x.X}, y = {y.X}, Objective = {m.objVal}")