function [x,n] = jacobi(A,b,x0,tol,times)
D = diag(diag(A));%求A的对角矩阵
L = -tril(A,-1);  %求A的下三角矩阵
U = -triu(A,1);   %求A的上三角矩阵
G = D\(L+U);
g = D\b;
x = G*x0+g;
n = 1;%迭代次数
while norm(x-x0)>=tol%2范数
    x0 = x;
    x = G*x0+g;
    n = n+1;
    if(n>=times)
        disp('too many iteration times!')
        return;
    end
end
end