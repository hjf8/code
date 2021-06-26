function [A]=mylu(A)
A=[10 20 30;20 45 80;30 80 171];
n=length(A);
for j=1:n-1
    A(j+1:n,j)=A(j+1:n,j)/A(j,j);
    A(j+1:n,j+1:n)=A(j+1:n,j+1:n)-A(j+1:n,j)*A(j,j+1:n);
end

%A=[1,1,1;2,3,4;5,6,7];
%LU=mylu(A);
%L=eye(length(A))+tril(A,-1);
%U=triu(A)