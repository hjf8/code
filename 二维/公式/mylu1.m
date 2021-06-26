function [L U]=mylu1(A)
%A²»ÊÇ·½Õó
%A=[1 2;3 4;5 6];
[n,m]=size(A);
for j=1:n-1
    A(j+1:n,j)=A(j+1:n,j)/A(j,j);
    A(j+1:n,j+1:m)=A(j+1:n,j+1:m)-A(j+1:n,j)*A(j,j+1:m);
end
if m>n
    L=eye(n)+tril(A(1:n,1:n),-1);
    U=triu(A);
else
    L=eye(n,m)+tril(A,-1);
    U=triu(A(1:m,1:m));
end
L;U;