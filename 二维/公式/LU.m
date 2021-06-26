function [L U]=LU(A,r)
A=[1,1,1,2;3,2,3,4;6,5,8,7;10,3,5,9];
r=2;
n=length(A);
[A11]=mylu(A(1:r,1:r));
A21=A(r+1:n,1:r);A12=A(1:r,r+1:n);
L11=eye(length(A11))+tril(A11,-1);U11=triu(A11);
L21=A21/U11;U12=L11\A12;
A22=A(r+1:n,r+1:n)-A21/A11*A12;
[A22]=mylu(A22);
L22=eye(length(A22))+tril(A22,-1);U22=triu(A22);
L11;L21;L22;U11;U12;U22;
L=[L11,zeros(r);L21,L22]
U=[U11,U12;zeros(r),U22]


