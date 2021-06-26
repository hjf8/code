function [x,n] = guaseidel(A,b,x0,tol,times)
D = diag(diag(A));%求A的对角矩阵
L = -tril(A,-1);%求A的下三角矩阵,不带对角线
U = -triu(A,1);%求A的上三角矩阵
G = (D-L)\U;
g = (D-L)\b;
x = G*x0+g;
n=1;  %迭代次数
while norm(x-x0)>=tol
    x0 = x;
    x = G*x0+g;
    n = n+1;
    if(n>=times)
        disp('too many iteration times!')
        return;
    end
end
end

