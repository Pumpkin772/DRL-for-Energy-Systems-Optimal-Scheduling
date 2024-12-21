import pyomo.environ as pyo
# 创建一个模型
model = pyo.ConcreteModel()

# 定义变量
model.x = pyo.Var(domain=pyo.NonNegativeReals)
model.y = pyo.Var(domain=pyo.NonNegativeReals)

# 添加目标函数（这里我们选择最大化x+y）
model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.maximize)

# 添加不可能满足的约束条件
# 这个约束条件要求x和y同时大于1和小于0，这是不可能的
model.con1 = pyo.Constraint(expr=model.x + model.y <= 0)
model.con2 = pyo.Constraint(expr=model.x >= 1)
model.con3 = pyo.Constraint(expr=model.y >= 1)

# 尝试求解模型
solver = pyo.SolverFactory('gurobi')  # 使用glpk求解器
results = solver.solve(model, tee=False)  # tee=True表示打印求解过程

# 检查求解状态
# if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.infeasible):
#     print('No solution found due to infeasibility.')
# elif results.solver.termination_condition == pyo.TerminationCondition.unbounded:
#     print('Problem is unbounded.')
# else:
#     print('Solver did not find an optimal solution.')