class Matrix:
    def __init__(self, n,m):
        self = []
        self.n = n
        self.m = m
        for i in range(n):
            self.append([0]*m)

    def __mul__(self,B):
        C = Matrix(self.n,B.m)
        
        for i in range(self.n):
            for j in range(B.m):
                for k in range(self.m):
                    C.matrix[i][j] += self.matrix[i][k]*B.matrix[k][j]
        return C
            
            
        
