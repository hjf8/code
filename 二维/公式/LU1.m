function [L U]=LU1(A,r)
A=[1 2 3;4 5 6;7 8 9];
[n,m]=size(A);
r=2;
A11=A(1:r,1:r);
[L11 U11]=mylu1(A11);
A21=A(r+1:n,1:r);A12=A(1:r,r+1:m);A22=A(r+1:n,r+1:m);
U12=L11\A12;L21=A21/U11;
H=A22-A21/A11*A12;
[L22 U22]=mylu1(H);
L=[L11,zeros(r,m-r);L21,L22];
U=[U11,U12;zeros(n-r,r),U22];
L,U

