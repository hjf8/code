function poisson1D(a,b,n)
h=(b-a)/n
x=[a:h:b];%以h为步长的点
A(1:n+1,1:n+1)=0;
A(1,1) =  1.0;
A(n+1,n+1) =  1.0;
R(1,1) = 0.0;
R(n+1,1)=0.0;
for i=2:n
   R(i,1)=sin(x(i))*h^2;
   A(i,i)=2;
   A(i,i-1)=-1;
   A(i,i+1)=-1;
end
size(A);
size(R);
%u = A\R；
%u=gauss(A,R);
x0=zeros(size(A,1),1);%以矩阵A的行数为x0的行数
%[u,n]=jacobi(A,R,x0,1.0e-6,1000000)；
[u]=Gauss2(A,R,x0);
%n
%[u, n] = CG(A,R,x0,1.0e-12,100000);
%n
plot(x,u)

end