from pyparsing import Or


def AND_gate(x1,x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.6
    result = w1*x1 + w2*x2 +b
    if result <= 0:
        return 0
    else:
        return 1

def NAND_gate(x1,x2):
    w1 = -0.5
    w2 = -0.5
    b = 0.6
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
    
def OR_gate(x1,x2):
    w1 = 0.6
    w2 = 0.6
    b = -0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
    
def XOR_gate(x1,x2):
    x_nand = NAND_gate(x1,x2)
    x_or = OR_gate(x1,x2)
    return AND_gate(x_nand,x_or)
    

print("AND_gate(0, 0), AND_gate(0, 1), AND_gate(1, 0), AND_gate(1, 1) : ",AND_gate(0, 0), AND_gate(0, 1), AND_gate(1, 0), AND_gate(1, 1))
print("NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1)",NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1))
print("OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1)",OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1))
print("XOR_gate(0, 0), XOR_gate(0, 1), XOR_gate(1, 0), XOR_gate(1, 1)",XOR_gate(0, 0), XOR_gate(0, 1), XOR_gate(1, 0), XOR_gate(1, 1))